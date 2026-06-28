
import numpy as np
import ruamel.yaml as yaml
import math
from llama import Llama
from engine.step_interpreters import register_step_interpreters, parse_step

from tools.image2text import I2T_model
from prompts.intermediate_result import INTERMEDIATE_FUNC



image2text = I2T_model()

LLM_config_path = 'configs/LLM_config.yaml'
LLM_config = yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

ckpt_dir = LLM_config['LLaMA']['ckpt_dir_path']
tokenizer_path = LLM_config['LLaMA']['tokenizer_path']
temperature = LLM_config['LLaMA']['temperature']
top_p = LLM_config['LLaMA']['top_p']
max_seq_len = LLM_config['LLaMA']['max_seq_len']
max_gen_len = LLM_config['LLaMA']['max_gen_len']
max_batch_size = LLM_config['LLaMA']['max_batch_size']

llama_generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


def concent_location_answer(input):
    index_s = input.find('\n\n')
    if index_s > 0:
        return input[:index_s + 1]
    else:
        return input


def concent_location(input):
    index_s = input.find('\n\n')
    if index_s > 2:
        return input[:index_s + 1]
    else:
        return input


class Program:
    def __init__(self, prog_str, init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self, task='gqa'):
        self.step_interpreters = register_step_interpreters(task, llama_generator)
        self.task = task

    def execute_step(self, prog_step):#, is_face):
        step_name = parse_step(prog_step.prog_str, partial=True)['step_name']
        # 执行
        if step_name == "SELECT":
            return self.step_interpreters[step_name].execute(prog_step)#, is_face)  # 执行模块
        elif step_name == 'SEG':
            return self.step_interpreters[step_name].execute(prog_step, query_name=self.seg_query)  # 执行模块
        else:
            return self.step_interpreters[step_name].execute(prog_step)  # 执行模块


    def execute(self, prog, init_state):# ,is_face=False):
        word_probability = []


        if 'SEG' in prog:
            select_list=self.search_select(prog,init_state)
            self.seg_query=select_list[0]['query']

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            print('新加的-进入了assert中断判断==============')
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        extra_out = None
        real_loc = None

        for prog_step in prog_steps:
            print('新加的-开始执行子步骤每一步的程序了==============')
            # 执行子步骤的时候，这点出现了错误
            step_name = parse_step(prog_step.prog_str, partial=True)['step_name']

            if step_name == 'VQA':
                step_output, word_probability = self.execute_step(prog_step)#, is_face)  # 返回两个变量
            else:
                step_output = self.execute_step(prog_step)  # 执行模块，返回一个变量

            if self.task == 'knowtag':
                if prog_step.prog_str.find("CLASSIFY") > -1:
                    extra_out = step_output.copy()
                if prog_step.prog_str.find("LOC") > -1:
                    real_loc = step_output

        return step_output, prog.state, extra_out, real_loc, word_probability

    def parse_vqa(self, prog_step, i):
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name'] == 'VQA':
            args = parse_result['args']
            img_var = args['image']
            question = eval(args['question'])
            output_var = parse_result['output_var']

            return dict(img_var=img_var, question=question, output_var=output_var, step=i)
        else:
            return None

    def search_vqa(self, prog, init_state):
        vqa_list = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:
            step_var = self.parse_vqa(prog_step, step)
            if step_var != None:
                vqa_list.append(step_var)
            step = step + 1

        return vqa_list

    def parse_loc(self, prog_step, i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name'] == 'LOC':
            args = parse_result['args']
            img_var = args['image']
            object1 = args['object']
            output_var = parse_result['output_var']

            return dict(img_var=img_var, object=object1, output_var=output_var, step=i)
        else:
            return None

    def search_loc(self, prog, init_state):
        vqa_list = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:
            step_var = self.parse_loc(prog_step, step)
            if step_var != None:
                vqa_list.append(step_var)
            step = step + 1

        return vqa_list

    def parse_select(self, prog_step, i, mod="SELECT"):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if mod == "SELECT":
            if parse_result['step_name'] == 'SELECT':
                args = parse_result['args']
                img_var = args['image']
                object1 = args['object']
                query = args['query']
                category = args['category']
                output_var = parse_result['output_var']

                return dict(img_var=img_var, object=object1, query=query, category=category, output_var=output_var,
                            step=i)
            else:
                return None
        elif mod == "REPLACE":
            if parse_result['step_name'] == 'REPLACE':
                args = parse_result['args']
                img_var = args['image']
                object1 = args['object']
                query = args['query']
                category = args['category']
                output_var = parse_result['output_var']

                return dict(img_var=img_var, object=object1, query=query, category=category, output_var=output_var,
                            step=i)
            else:
                return None

    def search_select(self, prog, init_state, mod="SELECT"):
        select_list = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:

            step_var = self.parse_select(prog_step, step, mod=mod)

            if step_var != None:
                select_list.append(step_var)
            step = step + 1

        return select_list

    def parse_seg(self, prog_step, i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name'] == 'SEG':
            args = parse_result['args']
            img_var = args['image']
            output_var = parse_result['output_var']

            return dict(img_var=img_var, output_var=output_var, step=i)
        else:
            return None

    def search_seg(self, prog, init_state):
        seg_list = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:
            step_var = self.parse_select(prog_step, step)
            if step_var != None:
                seg_list.append(step_var)
            step = step + 1

        return seg_list

    def parse_replace(self, prog_step, i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)

        if parse_result['step_name'] == 'REPLACE':
            args = parse_result['args']
            img_var = args['image']
            object1 = args['object']
            query = args['prompt']
            # category = args['category']
            output_var = parse_result['output_var']

            return dict(img_var=img_var, object=object1, query=query, output_var=output_var, step=i)
        else:
            return None

    def search_replace(self, prog, init_state):
        select_list = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:

            step_var = self.parse_replace(prog_step, step)

            if step_var != None:
                select_list.append(step_var)
            step = step + 1

        return select_list

    def parse_list(self, prog_step, i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name'] == 'LIST':
            args = parse_result['args']
            query = args['query']
            list_max = args['max']
            output_var = parse_result['output_var']

            return dict(query=query, list_max=list_max, output_var=output_var, step=i)
        else:
            return None

    def search_list(self, prog, init_state):
        list_list = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:
            step_var = self.parse_list(prog_step, step)
            if step_var != None:
                list_list.append(step_var)
            step = step + 1

        return list_list

    def parse_classify(self, prog_step, i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name'] == 'CLASSIFY':
            args = parse_result['args']
            image = args['image']
            object1 = args['object']
            categories = args['categories']
            output_var = parse_result['output_var']

            return dict(image=image, object=object1, categories=categories, output_var=output_var, step=i)
        else:
            return None

    def search_classify(self, prog, init_state):
        list_classify = []

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        step = 1
        for prog_step in prog_steps:
            step_var = self.parse_list(prog_step, step)
            if step_var != None:
                list_classify.append(step_var)
            step = step + 1

        return list_classify

    def update_vqavisualmodel(self, model_name, correct, data):
        print('------------------start update the visual model:', model_name)
        self.step_interpreters[model_name].update(correct, data)

    def update_locvisualmodel(self, model_name, data):
        print('------------------start update the LOC visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_selectvisualmodel(self, model_name, data):
        print('------------------start update the SELECT visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_segvisualmodel(self, model_name, data):
        print('------------------start update the SEG visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_replacevisualmodel(self, model_name, data):
        print('------------------start update the REPLACE visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_classifyvisualmodel(self, model_name, result, category_name):
        print('------------------start update the CLASSIFY visual model:', model_name)
        self.step_interpreters[model_name].update(result, category_name)


class ProgramGenerator():
    def __init__(self, subquestion_prompter, program_prompter, temperature=0.7, top_p=0.5, prob_agg='mean'):

        self.subquestion_prompter = subquestion_prompter  # 就是这个函数了create_subquestion_prompt()
        self.program_prompter = program_prompter  # 就是这个函数了create_subquestion_prompt()

        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self, response):
        eos = '<|endoftext|>'
        for i, token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token == eos:
                break

        if self.prob_agg == 'mean':
            agg_fn = np.mean
        elif self.prob_agg == 'sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    # 下面四个函数都是新加的
    def _is_pure_newline_token(self,tok: str) -> bool:
        # tok 只要包含换行，且去掉换行后全是空白，就认为是“纯换行token”
        # 例如 "\n" "\n\n" "\n   " 都算
        return tok is not None and ("\n" in tok) and (tok.replace("\n", "").strip() == "")
    # 这个函数是去除掉中间多余的换行符
    def _clean_tokens_for_line_ppl(self,tokens, logprobs):
        # 1) 对齐长度
        n = min(len(tokens), len(logprobs))
        tokens = tokens[:n]
        logprobs = logprobs[:n]

        # 2) 去掉开头纯换行（避免第一行空行 -> nan）
        while tokens and self._is_pure_newline_token(tokens[0]):
            tokens = tokens[1:]
            logprobs = logprobs[1:]

        # 3) 去掉结尾纯换行（避免最后空行 -> nan）
        while tokens and self._is_pure_newline_token(tokens[-1]):
            tokens = tokens[:-1]
            logprobs = logprobs[:-1]

        # 4) 折叠中间连续换行："\n\n" -> "\n"
        new_toks = []
        new_lps = []
        prev_was_newline = False
        for tok, lp in zip(tokens, logprobs):
            if self._is_pure_newline_token(tok):
                # 如果前一个保留下来的也是换行，那么这个换行会造成“空行” -> 跳过
                if prev_was_newline:
                    continue
                new_toks.append(tok)
                new_lps.append(lp)
                prev_was_newline = True
            else:
                new_toks.append(tok)
                new_lps.append(lp)
                prev_was_newline = False

        # 再保险：清掉尾部换行（可能折叠后又露出来）
        while new_toks and self._is_pure_newline_token(new_toks[-1]):
            new_toks.pop()
            new_lps.pop()

        return new_toks, new_lps

    # 将 tokens 切割一下，只要生成的子步骤
    def truncate_to_subquestion(self, tokens, logprobs):
        """
        只保留“子步骤(steps)”的 tokens/logprobs：
        - 遇到 '### Submission'（或其它非 Step 的段落头）就截断
        - 去掉首尾换行，避免空行导致 nan
        """
        n = min(len(tokens), len(logprobs))
        out_toks, out_lps = [], []

        # 这里保留你 truncate_to_code 的 prev1/prev2 风格，用来判断“段落切换处”
        prev2, prev1 = None, None

        for i, (tok, lp) in enumerate(zip(tokens[:n], logprobs[:n])):
            # ===== 1) 检测截断标记：### Submission =====
            # token 可能是 '##' + '#' + 'Sub' + 'mission'，所以用窗口拼接判断
            window = "".join(tokens[i: min(n, i + 8)])  # 8 足够覆盖 "### Submission"
            is_submission_heading = ("Submission" in window)

            # 额外保险：有时模型会在 steps 后接 "Program:" 或 "Question:"，也可以截断
            is_program_heading = ("Program:" in window) or (tok.strip() == "Program")
            is_question_heading = ("Question:" in window) or (tok.strip() == "Question")

            # 触发截断：通常这些标题前会有空行（至少一个 '\n'）
            if (is_submission_heading or is_program_heading or is_question_heading):
                if prev1 is not None and ("\n" in prev1):
                    # 去掉末尾多余换行，避免产生空行 -> nan
                    while out_toks and ("\n" in out_toks[-1]):
                        out_toks.pop()
                        out_lps.pop()
                    break

            out_toks.append(tok)
            out_lps.append(lp)
            prev2, prev1 = prev1, tok

        # ===== 2) 去掉开头换行（避免第一行空行 -> nan）=====
        while out_toks and ("\n" in out_toks[0]):
            out_toks.pop(0)
            out_lps.pop(0)

        # ===== 3) 去掉结尾换行（避免最后空行 -> nan）=====
        while out_toks and ("\n" in out_toks[-1]):
            out_toks.pop()
            out_lps.pop()

        return out_toks, out_lps

    # 将 tokens 切割一下，只要生成的代码
    def truncate_to_code(self,tokens, logprobs):
        """
        只保留“代码区”的 tokens/logprobs：
        遇到模式：\\n \\n Question（也就是代码结束后开始自然语言说明）就截断。
        同时去掉首尾的纯换行，避免空行导致 nan。
        """
        n = min(len(tokens), len(logprobs))
        out_toks = []
        out_lps = []

        prev2 = None
        prev1 = None

        for tok, lp in zip(tokens[:n], logprobs[:n]):
            # 触发截断：前面两个 token 都包含换行，并且当前 token 是 Question（或以 Question 开头）
            is_question = (tok.strip() == "Question") or tok.strip().startswith("Question")
            if is_question and prev1 is not None and ("\n" in prev1) and prev2 is not None and ("\n" in prev2):
                # 把已经写入的末尾两个换行也去掉（它们会形成一个空行 -> nan）
                if len(out_toks) >= 2 and ("\n" in out_toks[-1]) and ("\n" in out_toks[-2]):
                    out_toks = out_toks[:-2]
                    out_lps = out_lps[:-2]
                break

            out_toks.append(tok)
            out_lps.append(lp)
            prev2, prev1 = prev1, tok

        # 去掉开头的换行（避免第一行空行 -> nan）
        while out_toks and ("\n" in out_toks[0]):
            out_toks.pop(0)
            out_lps.pop(0)

        # 去掉结尾的换行（避免最后一行空行 -> nan）
        while out_toks and ("\n" in out_toks[-1]):
            out_toks.pop()
            out_lps.pop()

        return out_toks, out_lps

    # 新加的用来计算llama模型生成的内容的每行的 PPL
    def calc_line_ppls(self,token_strs, logprobs, exclude_newline_token=True):
        """
        token_strs: List[str]  每个 token 的文本（decode 后）
        logprobs:   List[float] 每个 token 的 log p（<=0），与 token_strs 对齐
        返回：line_ppls, overall_ppl
        """
        n = min(len(token_strs), len(logprobs))
        line_ppls = []
        cur_sum = 0.0  # 累积 log p
        cur_cnt = 0  # token 数

        all_sum = 0.0
        all_cnt = 0

        for tok, lp in zip(token_strs[:n], logprobs[:n]):
            if lp is None:
                continue

            # 是否包含换行（有些 tokenizer 可能一个 token 里含 '\n'）
            has_nl = ("\n" in tok)

            # 是否把“换行 token”的 logprob 计入上一行（通常不计）
            if not (exclude_newline_token and has_nl):
                cur_sum += lp
                cur_cnt += 1
                all_sum += lp
                all_cnt += 1

            if has_nl:
                # 结算当前行
                if cur_cnt > 0:
                    line_ppls.append(math.exp(-cur_sum / cur_cnt))
                else:
                    line_ppls.append(float("nan"))  # 空行
                cur_sum, cur_cnt = 0.0, 0

        # 最后一行没有换行也要结算
        if cur_cnt > 0:
            line_ppls.append(math.exp(-cur_sum / cur_cnt))

        overall_ppl = math.exp(-all_sum / all_cnt) if all_cnt > 0 else float("inf")
        return line_ppls, overall_ppl

    def generate(self, inputs):
        import math  # 确保引入 math

        print('\n\n\n')

        # ================= 1. 生成子问题 =================
        subquestion_prompt, subq_correct_index, sub_failed_index = self.subquestion_prompter(
            inputs,
            dict(question=inputs['question']),
            index=True
        )
        print('------------------subquestion_prompt start------------------')
        print(subquestion_prompt) # 调试时可以注释掉，避免刷屏
        print('------------------subquestion_prompt end------------------')

        subq_ppl = {}
        prog_ppl = {}
        subquestion = ''
        try:
            # 必须包装成列表！
            subquestion_prompt_list = [subquestion_prompt]

            response = llama_generator.text_completion(
                subquestion_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,  # 开启 PPL 计算
            )

            # ==============================这里开始改--计算子步骤 PPL 的值=======================================
            # 获取结果
            result = response[0]
            print("subquestion_response的值：", response)
            # 得到生成的每一个 tokens 的 logprobs
            raw_logprobs = result.get('logprobs', [])
            # 得到生成的每一个 tokens
            token_strs = result.get('tokens', [])
            subquestion = response[0]['generation']
            print("最终生成的子步骤:\n", subquestion)

            # # 裁剪一下生成的tokens，只留下代码部分，使计算的时候只算每行代码的PPL
            subq_strs, subq_logprobs = self.truncate_to_subquestion(token_strs, raw_logprobs)

            # 去掉句子中间多余的换行
            token_strs, raw_logprobs = self._clean_tokens_for_line_ppl(subq_strs, subq_logprobs)

            if not token_strs:
                print("警告：没有token生成，直接跳过计算！！！！！！！")
                line_ppls, overall_ppl = [], 0.0
            else:
                line_ppls, overall_ppl = self.calc_line_ppls(token_strs, raw_logprobs)

            print("生成子步骤的每行 PPL:", [None if (isinstance(x, float) and math.isnan(x)) else round(x, 4) for x in line_ppls])
            print(f"生成子步骤的整段 PPL: {overall_ppl:.4f}")

            subq_ppl = {"line_ppls": line_ppls,"overall_ppl": overall_ppl}  # 保留四位小数

        # ============================== 这里结束 =======================================
            subquestion = response[0]['generation']
            subquestion = concent_location(subquestion)
            subquestion = subquestion.lstrip().rstrip('\n')

        except Exception as e:
            print(f"Error generating subquestion: {e}")
            subquestion = 'some errors'

        # ================= 2. 生成程序 =================
        program_prompt, prog_correct_index, prog_failed_index = self.program_prompter(
            dict(question=inputs['question'], subquestion=subquestion),
            index=True
        )
        print('------------------program_prompt start------------------')
        print(program_prompt)
        print('------------------program_prompt end------------------')

        prog = ''
        try:
            # 【关键修改】你的模型 Batch Size 限制为 1，所以这里不能乘 5
            # 必须是一个只包含一个字符串的列表
            program_prompt_list = [program_prompt]

            print(f"!!! [DEBUG] Temperature: {temperature}, Top_p: {top_p}")

            # 调用模型
            program_response = llama_generator.text_completion(
                program_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,  # 开启 PPL 计算
            )

# ==============================这里开始改--计算程序的 PPL 的值=======================================
            # 获取结果
            result = program_response[0]
            print("program_response的值：",program_response)
            generated_text = result['generation']
            # 得到生成的每一个 tokens 的 logprobs
            raw_logprobs = result.get('logprobs', [])
            # 得到生成的每一个 tokens
            token_strs = result.get('tokens', [])

            prog = generated_text

            # 后处理
            prog = concent_location(prog)
            prog = prog.lstrip('\n').rstrip('\n')

            print("最终生成的程序:\n", prog)

            # --- 计算 PPL ---

            # 裁剪一下生成的tokens，只留下代码部分，使计算的时候只算每行代码的PPL
            token_strs, raw_logprobs = self.truncate_to_code(token_strs, raw_logprobs)

            # 去掉句子中间多余的换行
            code_tokens, code_logprobs = self._clean_tokens_for_line_ppl(token_strs, raw_logprobs)


            if not token_strs:
                print("警告：没有token生成，直接跳过计算！！！！！！！")
                line_ppls, overall_ppl = [], 0.0
            else:
                line_ppls, overall_ppl = self.calc_line_ppls(code_tokens, code_logprobs)

            print("生成子程序的每行 PPL:",
                  [None if (isinstance(x, float) and math.isnan(x)) else round(x, 4) for x in line_ppls])
            print(f"生成子程序的整段 PPL: {overall_ppl:.4f}")

            prog_ppl = {"line_ppls": line_ppls, "overall_ppl": overall_ppl}  # 保留四位小数



        except Exception as e:
            print(f"Error generating program: {e}")
            import traceback
            traceback.print_exc()
            prog = 'some errors'

        # print("最终程序:\n", prog)
        # ==============================这里结束=======================================
        index = [subq_correct_index, sub_failed_index, prog_correct_index, prog_failed_index]
        return subquestion, prog, index, prog_ppl, subq_ppl



class PartReflectioner():
    def __init__(self, part_reflection_prompter_step, part_reflection_prompter_stepbystep,
                 part_reflection_prompter_interrupt, temperature=0.7, top_p=0.5, prob_agg='mean'):
        self.part_reflection_prompter_step = part_reflection_prompter_step
        self.part_reflection_prompter_stepbystep = part_reflection_prompter_stepbystep
        self.part_reflection_prompter_interrupt = part_reflection_prompter_interrupt
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

        self.intermediate = INTERMEDIATE_FUNC[LLM_config['Task_type']]

    def compute_prob(self, response):
        eos = '<|endoftext|>'
        for i, token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token == eos:
                break

        if self.prob_agg == 'mean':
            agg_fn = np.mean
        elif self.prob_agg == 'sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    def analyze_interrupt(self, inputs):

        reflection_prompt = self.part_reflection_prompter_interrupt(
            dict(question=inputs['question'], human_feedback=inputs['human_feedback'],
                 subquestion=inputs['subquestion'], program=inputs['program']))

        print('------------------interrupt reflection_prompt start------------------')
        print(reflection_prompt.lstrip('\n').rstrip('\n'))
        print('------------------interrupt reflection_prompt end------------------')

        try:
            reflection_prompt_list = []
            reflection_prompt_list.append(reflection_prompt)
            reflection = llama_generator.text_completion(
                reflection_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            reflection = reflection[0]['generation']
            reflection = concent_location(reflection)
            reflection = reflection.lstrip('\n').rstrip('\n')
        except:
            reflection = 'some errors'

        return reflection

    def analyze_step(self, inputs, prog_state):

        intermediate_output = self.intermediate(prog_state)
        reflection_prompt = self.part_reflection_prompter_step(
            dict(question=inputs['question'], human_feedback=inputs['human_feedback'],
                 subquestion=inputs['subquestion'], program=inputs['program'], intermediate_output=intermediate_output),
            prog_state)

        print('------------------step reflection_prompt start------------------')
        print(reflection_prompt.lstrip('\n').rstrip('\n'))
        print('------------------step reflection_prompt end------------------')
        print()
        print()
        print()

        try:
            reflection_prompt_list = []
            reflection_prompt_list.append(reflection_prompt)

            reflection = llama_generator.text_completion(
                reflection_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            reflection = reflection[0]['generation']
            reflection = concent_location(reflection)
            reflection = reflection.lstrip('\n').rstrip('\n')
        except:
            reflection = 'some errors'

        return reflection

    def analyze_stepbystep(self, inputs, prog_state):

        reason_find = False
        reason = None
        error_step = None

        intermediate_output = self.intermediate(prog_state)
        d_subquestion = inputs['subquestion']
        d_subquestion = d_subquestion.split('\n')
        total_number_step = len(d_subquestion)

        for i in range(total_number_step):
            reflection_prompt = self.part_reflection_prompter_stepbystep(
                dict(question=inputs['question'], human_feedback=inputs['human_feedback'],
                     subquestion=inputs['subquestion'], program=inputs['program'],
                     intermediate_output=intermediate_output), prog_state, i + 1)

            print()
            print()
            print()
            print('------------------stepbystep reflection_prompt start------------------')
            print(reflection_prompt.lstrip('\n').rstrip('\n'))
            print('------------------stepbystep reflection_prompt end------------------')
            print()
            print()
            print()

            try:
                reflection_prompt_list = []
                reflection_prompt_list.append(reflection_prompt)

                print()
                print()
                print()
                print(
                    "新加的-reflection_prompt_list提示词列表的值(这个才是真正交给llama模型的提示词)=========================================================================================================================================================================================================")
                # print(reflection_prompt_list)
                # 替换原来的 print(reflection_prompt_list)
                for prompt in reflection_prompt_list:
                    print(prompt)
                print()
                print()
                print()

                # list = ["你好"]
                # 上面生成的提示词reflection_prompt_list没问题，下面生成的reflection有问题
                # 说明是下面这个llama_generator.text_completion有问题( 根据提示词输出reflection的时候出问题了)
                # 也可能是提示词有问题
                reflection = llama_generator.text_completion(
                    reflection_prompt_list,
                    # list,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p
                )

                print()
                print()
                print()
                print(
                    "新加的-正在找slow_analysis出错的原因=======================================================================================================================================================================================================")
                print("截取前reflection的值为：\n", reflection)
                print()
                print()
                print()
                reflection = reflection[0]['generation']
                reflection = concent_location(reflection)
                reflection = reflection.lstrip('\n').rstrip('\n')
            except:
                reflection = 'some errors'

            if 'yes' in reflection or 'Yes' in reflection:
                continue
            else:
                print()
                print()
                print()
                print("新加的-正在找slow_analysis出错的原因\n截取后reflection的值为：\n", reflection)
                print()
                print()
                print()

                reason_find = True
                no_location = reflection.find('No')
                reason = reflection[no_location + 3:]
                error_step = 'Error is in Step' + str(i + 1) + '. '
                break

        return reason_find, reason, error_step


def reason_locate(reflection):
    reason_index = reflection.find('Reason:')

    if reason_index < 2:
        reason_end_index = reflection.find('\n\n')
        return reflection[:reason_end_index], reflection[:reason_end_index]
    else:

        location = reflection[:reason_index]
        reason_end_index = reflection.find('\n\n')
        reason = reflection[reason_index + 7:reason_end_index]

        return location, reason


class Program_React_Generator():
    def __init__(self, subquestion_prompter, program_prompter, subquestion_react_prompter, program_react_prompter,
                 part_inference_prompter, temperature=0.7, top_p=0.5, prob_agg='mean'):

        self.subquestion_prompter = subquestion_prompter
        self.program_prompter = program_prompter
        self.subquestion_react_prompter = subquestion_react_prompter
        self.program_react_prompter = program_react_prompter

        self.answer_inference_prompter = part_inference_prompter

        self.intermediate = INTERMEDIATE_FUNC[LLM_config['Task_type']]

        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self, response):
        eos = '<|endoftext|>'
        for i, token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token == eos:
                break

        if self.prob_agg == 'mean':
            agg_fn = np.mean
        elif self.prob_agg == 'sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    def generate(self, inputs):

        errorlocation = inputs['errorlocation']

        if ('subquestion' in errorlocation) or ('SubQuestion' in errorlocation) or ('Subquestion' in errorlocation):
            subquestion_prompt = self.subquestion_react_prompter(inputs)
        else:
            subquestion_prompt = self.subquestion_prompter(inputs, dict(question=inputs['question'],
                                                                        pre_subq=inputs['pre_subq']))
        print()
        print()
        print()
        print('------------------react_subquestion_prompt start------------------')
        print(subquestion_prompt)
        print('------------------react_subquestion_prompt end------------------')
        print()
        print()
        print()

        try:
            subquestion_prompt_list = []
            subquestion_prompt_list.append(subquestion_prompt)
            response = llama_generator.text_completion(
                subquestion_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            subquestion = response[0]['generation']
            subquestion = concent_location(subquestion)
            subquestion = subquestion.lstrip('\n').rstrip('\n')
        except:
            subquestion = 'some errors'

        if 'program' in errorlocation:
            program_prompt = self.program_react_prompter(
                dict(question=inputs['question'], subquestion=inputs['subquestion'], program=inputs['program'],
                     errorlocation=inputs['errorlocation'], reason=inputs['reason'], newsubquestion=subquestion))
        else:
            # 修改前
            # program_prompt=self.program_prompter(dict(question=inputs['question'], subquestion=subquestion),dict(question=inputs['question'], pre_subq=inputs['pre_subq'], pre_prog=inputs['pre_prog']))
            program_prompt = self.program_prompter(dict(question=inputs['question'], subquestion=subquestion))

        print()
        print()
        print()
        print('新加的 ------------------react_program_提示词 开始------------------')
        print(program_prompt)
        print('新加的 ------------------react_program_提示词 结束------------------')
        print()
        print()
        print()

        try:
            program_prompt_list = []
            program_prompt_list.append(program_prompt)
            program_reresponse = llama_generator.text_completion(
                program_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            prog = program_reresponse[0]['generation']
            prog = concent_location(prog)
            prog = prog.lstrip('\n').rstrip('\n')
        except:
            prog = 'some errors'

        print()
        print()
        print()
        print('------------------新加的-react_subquestion_response start------------------')
        print(subquestion)
        print('------------------新加的-react_subquestion_response end------------------')
        print()
        print()
        print()
        print('------------------新加的-react_program_response start------------------')
        print(prog)
        print('------------------新加的-react_program_response end------------------')
        print()
        print()
        print()


        return subquestion, prog

    def answer_inference(self, inputs, prog_state):

        intermediate_output = self.intermediate(prog_state)
        inference_prompt = self.answer_inference_prompter(
            dict(question=inputs['question'], human_feedback=inputs['human_feedback'],
                 subquestion=inputs['subquestion'], program=inputs['program'], intermediate_output=intermediate_output,
                 errorlocation=inputs['errorlocation'], reason=inputs['reason']), prog_state)

        print('------------------answer_inference_prompt start------------------')
        print(inference_prompt.lstrip('\n').rstrip('\n'))
        print('------------------answer_inference_prompt end------------------')

        try:
            inference_prompt_list = []
            inference_prompt_list.append(inference_prompt)
            inference = llama_generator.text_completion(
                inference_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            inference = inference[0]['generation']
            inference = concent_location_answer(inference)
            inference = inference.lstrip('\n').rstrip('\n')
        except:
            inference = 'some errors'

        return inference