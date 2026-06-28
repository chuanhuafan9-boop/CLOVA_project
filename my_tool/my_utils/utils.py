
import numpy as np
import ruamel.yaml as yaml
import math
from llama import Llama
from engine.step_interpreters import register_step_interpreters, parse_step
from my_tool.prompts.prompt_engineering import create_regerate_subquestion_prompt_with_reflection, \
    create_regerate_program_prompt_with_reflection

from tools.image2text import I2T_model
from prompts.intermediate_result import INTERMEDIATE_FUNC
import re
import json



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

"""
extract_json_fields、_salvage_by_regex、_extract_list_field
_extract_string_field、_repair_json_string、_count_unescaped_quotes
_clean_text
这几个函数都是为了将模型生成的回复提取成 json 的形式
"""
def _clean_text(text):
    """
    预清洗：
    1. 去掉 markdown 代码块
    2. 去掉首尾空白
    3. 修正常见引号
    """
    if text is None:
        return ""

    text = str(text).strip()

    # 去掉 ```json ... ``` 或 ``` ... ```
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # 统一中英文引号
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    return text
def _count_unescaped_quotes(s):
    """
    统计未被反斜杠转义的双引号个数
    """
    return len(re.findall(r'(?<!\\)"', s))
def _repair_json_string(json_str):
    """
    对提取出来的 JSON 字符串做尽量温和的修复：
    1. 修复非法转义 \'
    2. 自动补开头 {
    3. 如果双引号数量是奇数，补一个 "
    4. 自动补 ] 和 }
    5. 去掉结尾多余逗号
    """
    json_str = json_str.strip()

    # 修复非法 JSON 转义：\' -> '
    json_str = json_str.replace(r"\'", "'")

    # 如果没有开头 {，补一个
    if not json_str.startswith("{"):
        json_str = "{" + json_str

    # 如果双引号数量是奇数，说明最后一个字符串大概率被截断了，补一个 "
    if _count_unescaped_quotes(json_str) % 2 == 1:
        json_str += '"'

    # 先补 ]
    left_bracket = json_str.count("[")
    right_bracket = json_str.count("]")
    if right_bracket < left_bracket:
        json_str += "]" * (left_bracket - right_bracket)

    # 再补 }
    left_brace = json_str.count("{")
    right_brace = json_str.count("}")
    if right_brace < left_brace:
        json_str += "}" * (left_brace - right_brace)

    # 去掉类似 {"a":1,} 这种结尾多余逗号
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json_str
def _extract_string_field(text, key, next_keys=None):
    """
    从半截 JSON 文本里，尽量抢救出字符串字段。
    适用于：
    "error_reason": "xxxx,
    "fix_strategy": "yyyy
    这种可能被截断的情况
    """
    if next_keys is None:
        next_keys = []

    start_pattern = rf'"{re.escape(key)}"\s*:\s*"'
    start_match = re.search(start_pattern, text, re.DOTALL)
    if not start_match:
        return ""

    start = start_match.end()
    tail = text[start:]

    # 优先找到下一个 key 的位置，作为当前字段结束
    next_positions = []
    for nk in next_keys:
        m = re.search(rf'",?\s*"{re.escape(nk)}"\s*:', tail, re.DOTALL)
        if m:
            next_positions.append(m.start())

    if next_positions:
        end = min(next_positions)
        value = tail[:end]
    else:
        # 没找到下一个 key，就取到末尾，再把末尾可能多出来的引号 / 右括号去掉
        value = tail
        value = re.sub(r'"\s*}\s*$', "", value)
        value = re.sub(r'"\s*$', "", value)
        value = re.sub(r'}\s*$', "", value)

    # 修复非法转义
    value = value.replace(r"\'", "'").strip()
    return value
def _extract_list_field(text, key):
    """
    从半截 JSON 文本里尽量提取列表字段，例如：
    "substep_ids": ["Step1", "Step2", "Step3"]
    即使缺右中括号，也尽量提取
    """
    pattern = rf'"{re.escape(key)}"\s*:\s*\[(.*?)(?:\]|\Z)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []

    inner = match.group(1)
    items = re.findall(r'"([^"]*)"', inner)
    return items
def _salvage_by_regex(text):
    """
    当 json.loads 失败后，用正则做兜底抢救。
    返回固定 schema，避免后续代码崩溃。
    """
    result = {
        "error_type": "",
        "error_layer": "",
        "substep_ids": [],
        "error_reason": "",
        "fix_strategy": ""
    }

    result["error_type"] = _extract_string_field(
        text,
        "error_type",
        next_keys=["error_layer", "substep_ids", "error_reason", "fix_strategy"]
    )

    result["error_layer"] = _extract_string_field(
        text,
        "error_layer",
        next_keys=["substep_ids", "error_reason", "fix_strategy"]
    )

    result["substep_ids"] = _extract_list_field(text, "substep_ids")

    result["error_reason"] = _extract_string_field(
        text,
        "error_reason",
        next_keys=["fix_strategy"]
    )

    result["fix_strategy"] = _extract_string_field(
        text,
        "fix_strategy",
        next_keys=[]
    )

    # 如果一个字段都没提出来，返回 None
    if (
        result["error_type"] == ""
        and result["error_layer"] == ""
        and result["substep_ids"] == []
        and result["error_reason"] == ""
        and result["fix_strategy"] == ""
    ):
        return None

    return result
def extract_json_fields(text):
    """
    从大模型输出中提取 JSON。
    处理这些异常情况：
    1. 缺开头 {
    2. 缺结尾 }
    3. 最后一个字符串被截断
    4. 非法转义 \'
    5. 外部包着 ```json
    6. json.loads 失败时按字段抢救
    """
    text = _clean_text(text)

    # 第一步：优先尝试提取 {...} 之间的内容
    match = re.search(r'\{.*\}', text, re.DOTALL)

    json_str = ""
    if match:
        json_str = match.group(0)
    else:
        # 如果没有完整的 {}，尝试从第一个 key 开始往后取
        fallback_match = re.search(r'"error_type"\s*:', text)
        if fallback_match:
            json_str = text[fallback_match.start():]
        else:
            # 再退一步：从第一个双引号开始到文本结尾
            fallback_match = re.search(r'".*', text, re.DOTALL)
            if fallback_match:
                json_str = fallback_match.group(0)

    if not json_str:
        print("未在文本中找到类似 JSON 的结构。")
        return None

    # 第二步：修复后尝试标准 JSON 解析
    repaired_json_str = _repair_json_string(json_str)

    try:
        data_dict = json.loads(repaired_json_str)
        return data_dict
    except json.JSONDecodeError as e:
        print(f"标准 JSON 解析失败: {e}")
        print("修复后的字符串是：")
        print(repaired_json_str)
        print("repr(repaired_json_str) =", repr(repaired_json_str))

    # 第三步：如果还是失败，就按字段抢救
    salvaged = _salvage_by_regex(repaired_json_str)
    if salvaged is not None:
        print("已启用正则兜底抢救，返回部分解析结果。")
        return salvaged

    print("正则兜底也失败，返回 None。")
    return None
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

        print('新加的-在execute_step里面执行生产的代码==============')
        return self.step_interpreters[step_name].execute(prog_step)  # 返回两个变量


    def execute(self, prog, init_state): #, is_face=False):
        word_probability = []

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


    def update_vqavisualmodel(self, model_name, correct, data):
        print('------------------start update the visual model:', model_name)
        self.step_interpreters[model_name].update(correct, data)

    def update_locvisualmodel(self, model_name, data):
        print('------------------start update the LOC visual model:', model_name)
        self.step_interpreters[model_name].update(data)



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

            # ==============================这里开始改=======================================
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

# ==============================这里开始改=======================================
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

    # 对生成的子步骤和程序进行截取得到只有子步骤和程序的结果
    def _normalize_text(self,text: str) -> str:
        """
        统一换行，去掉首尾空白。
        """
        if text is None:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()

    def extract_substeps(self,text: str) -> str:
        """
        从大模型输出中截取合法的子步骤文本。

        目标输出格式：
        Step1, ...
        Step2, ...
        ...
        StepN, Visualize results.

        处理逻辑：
        1. 只保留以 Step数字, 开头的行
        2. 从第一个 Step 行开始截取
        3. 遇到 'Visualize results.' 后停止
        4. 自动丢弃 Question / Error reason / Program 等杂项文本
        """
        text = self._normalize_text(text)
        if not text:
            return ""

        lines = text.split("\n")
        extracted = []
        started = False

        step_pattern = re.compile(r"^Step\d+,\s*")

        for raw_line in lines:
            line = raw_line.strip()

            # 找到第一个合法 Step 行才开始
            if step_pattern.match(line):
                started = True
                extracted.append(line)

                # 如果已经到可视化步骤，直接结束
                if "Visualize results." in line:
                    break

            elif started:
                # 已开始后，遇到非 Step 行，通常是脏文本
                # 这里直接跳过，不中断，防止中间混入空行或杂项
                continue

        # 如果没有提取到任何 Step 行，返回空字符串
        if not extracted:
            return ""

        # 二次清理：确保只保留连续 Step 行
        cleaned = []
        for line in extracted:
            if step_pattern.match(line):
                cleaned.append(line)

        return "\n".join(cleaned).strip()

    def extract_program(self,text: str) -> str:
        """
        从大模型输出中截取合法程序行。

        合法程序行格式：
        BOXi=...
        IMAGEi=...
        ANSWERi=...
        FINAL_RESULT=RESULT(var=...)

        处理逻辑：
        1. 只保留合法程序前缀行
        2. 从第一条合法程序行开始
        3. 遇到 FINAL_RESULT=RESULT(...) 后停止
        4. 自动过滤 Question / Error reason / Revised subquestion 等脏文本
        """
        text = self._normalize_text(text)
        if not text:
            return ""

        lines = text.split("\n")
        extracted = []
        started = False

        prog_pattern = re.compile(
            r"^(BOX\d+=|IMAGE\d+=|ANSWER\d+=|FINAL_RESULT=RESULT\(var=.*\))"
        )

        for raw_line in lines:
            line = raw_line.strip()

            if prog_pattern.match(line):
                started = True
                extracted.append(line)

                if line.startswith("FINAL_RESULT=RESULT"):
                    break

            elif started:
                # 已开始后，如果遇到非程序行，直接跳过
                continue

        if not extracted:
            return ""

        cleaned = []
        for line in extracted:
            if prog_pattern.match(line):
                cleaned.append(line)

        return "\n".join(cleaned).strip()

    # 后面regenerate_subq_and_prog_by_reflection、generate_type_of_error 这两个函数
    # 都是为了测试我的错误原因好不好加进来的
    def regenerate_subq_and_prog_by_reflection(self,question, init_subq, init_prog, error_reason, num_of_example):
        """
        这个函数是我自己加的，为了测试我的错误原因的好坏
        我要利用系统的错误原因来重新生成一个程序来
        看看是不是可以成功完成任务
        num_of_example: 表示的是例子的数量
        """
        regenerated_subquestion = ''
        regenerated_program = ''

        inputs = dict()
        inputs["question"] = question
        inputs["init_subq"] = init_subq
        inputs["init_prog"] = init_prog
        inputs["error_reason"] = error_reason

        # 使用自己定义的函数来生成 prompt
        num_of_example =num_of_example
        regenerate_subq_prompt = create_regerate_subquestion_prompt_with_reflection(inputs, num_of_example, method = "retrieval")
        #下面这个是成功了的一个 prompt
#         regenerate_subq_prompt = f"""
#     You need to act as a planner-reflector. Your goal is that, given a question, the initial planned sub-steps, the initial generated program, and an error reason describing the main mistake, you need to revise the subquestions so that the revised subquestions can correctly solve the original question.
#
# First, you will be given some correct revision examples.
#
# Question: Is the dog to the right of the chair both brown and small?
# Initial planned sub-steps:
# Step1, Locate the chair, and obtain bounding boxes of the chair.
# Step2, Crop the right part of the chair since the question is asking what is to the right of the chair. The bounding boxes are obtained in Step1.
# Step3, Try locate dog in the cropped image. The image is cropped in Step2.
# Step4, Count the number of bounding boxes. The bounding box is from Step3.
# Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
# Step6, Visualize results.
# Initial generated program:
# BOX0=LOC(image=IMAGE,object='chair')
# IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
# BOX1=LOC(image=IMAGE0,object='dog')
# BOX2=LOC(image=IMAGE0,object='dog')
# ANSWER0=COUNT(box=BOX1)
# ANSWER1=COUNT(box=BOX2)
# ANSWER2=EVAL(expr="'yes' if {{ANSWER0}} > 0 and {{ANSWER1}} > 0 else 'no'")
# FINAL_RESULT=RESULT(var=ANSWER2)
# Error reason:
# The old plan does not actually verify whether the dog is both brown and small. It only repeats dog detection and then uses count-based yes/no logic, so it ignores the required attributes.
# Revised subquestion:
# Step1, Locate the chair, and obtain bounding boxes of the chair.
# Step2, Crop the image region to the right of the chair, based on bounding boxes obtained in Step1.
# Step3, Ask whether the dog in the cropped image region from Step2 is both brown and small.
# Step4, Visualize results.
#
# Question: What material is the bag of the woman on the left?
# Initial planned sub-steps:
# Step1, Locate the woman, and obtain bounding boxes of the woman.
# Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
# Step3, Locate the bag in the cropped image region, and obtain bounding boxes of the bag.
# Step4, Count the number of bag bounding boxes, based on Step3.
# Step5, Determine whether the answer is 'leather' or 'cloth' by executing Python expression, based on the number from Step4.
# Step6, Visualize results.
# Initial generated program:
# BOX0=LOC(image=IMAGE,object='woman')
# IMAGE0=CROP(image=IMAGE,box=BOX0)
# BOX1=LOC(image=IMAGE0,object='bag')
# ANSWER0=COUNT(box=BOX1)
# ANSWER1=EVAL(expr="'leather' if {{ANSWER0}} > 0 else 'cloth'")
# FINAL_RESULT=RESULT(var=ANSWER1)
# Error reason:
# The old plan changes a material question into count-based fixed-choice logic, and it ignores the important constraint 'on the left'.
# Revised subquestion:
# Step1, Locate the woman on the left, and obtain bounding boxes of the woman.
# Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
# Step3, Ask what material the bag is in the cropped image region from Step2.
# Step4, Visualize results.
#
# Based on the correct revision examples, you need to revise the following planned subquestions according to the error reason.
#
# Rules:
# 1. Preserve the original question type.
# 2. Fix the main mistake described in the error reason.
# 3. Keep correct parts of the original plan when possible.
# 4. Remove redundant or duplicated steps.
# 5. The revised subquestions must be solvable by the available functions:
# LOC(image,object)
# COUNT(box)
# CROP(image,box)
# CROP_RIGHTOF(image,box)
# CROP_LEFTOF(image,box)
# CROP_BELOW(image,box)
# CROP_ABOVE(image,box)
# CROP_FRONTOF(image,box)
# CROP_BEHIND(image,box)
# VQA(image,question)
# EVAL(expr)
# RESULT(var)
# 6. Output only revised subquestions.
# 7. Every line must begin with Step1, Step2, Step3, ...
# 8. The final line must be Visualize results.
#
# Question: {question}
# Initial planned sub-steps:
# {init_subq}
# Initial generated program:
# {init_prog}
# Error reason:
# {error_reason}
# Revised subquestion:
#     """
        print("=========================根据出错原因重新生成子步骤的prompt-开始==============================")
        print( regenerate_subq_prompt)
        print("=========================根据出错原因重新生成子步骤的prompt-结束==============================")
        try:
            # 必须包装成列表！
            subquestion_prompt_list = [regenerate_subq_prompt]

            subq_response = llama_generator.text_completion(
                subquestion_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            # ==============================这里开始改=======================================
            # 获取结果
            result = subq_response[0]
            print("测试prompt回复的response的值：", subq_response)
            regenerated_subquestion = subq_response[0]['generation']
            regenerated_subquestion = self.extract_substeps(regenerated_subquestion)
            inputs["regenerated_subquestion"] = regenerated_subquestion


        except Exception as e:
            print("发生了错误，具体错误是:", e)
        print("测试 regenerate_subq_prompt 重新生成的子步骤是 ：\n", regenerated_subquestion)

        regenerate_prog_prompt = create_regerate_program_prompt_with_reflection(inputs, num_of_example, method = "retrieval")
        print("=========================根据出错原因重新生成程序的prompt-开始==============================")
        print( regenerate_prog_prompt)
        print("=========================根据出错原因重新生成程序的prompt-结束==============================")

#         regenerate_prog_prompt = f"""
# You need to act as a programmer-reflector. Your goal is that, given a function set, a question, the initial planned sub-steps, the initial generated program, an error reason describing the main mistake, and the revised subquestions, you need to generate a revised program that correctly solves the original question.
#
# Available functions are as follows.
# LOC: This function locates the queried region in an image. Definition: LOC(image,object). Input arguments: image and object's name. Output arguments: bounding boxes of the object.
# COUNT: This function counts the number of bounding boxes. Definition: COUNT(box). Input arguments: bounding boxes. Output arguments: number of the bounding boxes.
# CROP: This function crops an image region given co-ordinates of a bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# CROP_RIGHTOF: This function crops an image region to the right of a given bounding box. Definition: CROP_RIGHTOF(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# CROP_LEFTOF: This function crops an image region to the left of a given bounding box. Definition: CROP_LEFTOF(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# CROP_BELOW: This function crops an image region below a given bounding box. Definition: CROP_BELOW(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# CROP_ABOVE: This function crops an image region above a given bounding box. Definition: CROP_ABOVE(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# CROP_FRONTOF: This function crops an image region in front of a given bounding box. Definition: CROP_FRONTOF(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# CROP_BEHIND: This function crops an image region behind a given bounding box. Definition: CROP_BEHIND(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
# VQA: This function generates answers of questions based on a given image. Definition: VQA(image,question). Input arguments: image and question. Output arguments: answer.
# EVAL: This function executes Python expression in textual form to obtain the answer. Definition: EVAL(expr). Input arguments: Python expression in textual form. Output arguments: answer.
# RESULT: This function is finally used to visualize results in html. Definition: RESULT(var). Input arguments: image or text.
#
# Then, you will be given some correct revision examples.
#
# Question: Is the dog to the right of the chair both brown and small?
# Initial planned sub-steps:
# Step1, Locate the chair, and obtain bounding boxes of the chair.
# Step2, Crop the right part of the chair since the question is asking what is to the right of the chair. The bounding boxes are obtained in Step1.
# Step3, Try locate dog in the cropped image. The image is cropped in Step2.
# Step4, Count the number of bounding boxes. The bounding box is from Step3.
# Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
# Step6, Visualize results.
# Initial generated program:
# BOX0=LOC(image=IMAGE,object='chair')
# IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
# BOX1=LOC(image=IMAGE0,object='dog')
# BOX2=LOC(image=IMAGE0,object='dog')
# ANSWER0=COUNT(box=BOX1)
# ANSWER1=COUNT(box=BOX2)
# ANSWER2=EVAL(expr="'yes' if {{ANSWER0}} > 0 and {{ANSWER1}} > 0 else 'no'")
# FINAL_RESULT=RESULT(var=ANSWER2)
# Error reason:
# The old plan does not actually verify whether the dog is both brown and small. It only repeats dog detection and then uses count-based yes/no logic, so it ignores the required attributes.
# Revised subquestion:
# Step1, Locate the chair, and obtain bounding boxes of the chair.
# Step2, Crop the image region to the right of the chair, based on bounding boxes obtained in Step1.
# Step3, Ask whether the dog in the cropped image region from Step2 is both brown and small.
# Step4, Visualize results.
# Revised program:
# BOX0=LOC(image=IMAGE,object='chair')
# IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
# ANSWER0=VQA(image=IMAGE0,question='Is the dog both brown and small?')
# FINAL_RESULT=RESULT(var=ANSWER0)
#
# Question: What material is the bag of the woman on the left?
# Initial planned sub-steps:
# Step1, Locate the woman, and obtain bounding boxes of the woman.
# Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
# Step3, Locate the bag in the cropped image region, and obtain bounding boxes of the bag.
# Step4, Count the number of bag bounding boxes, based on Step3.
# Step5, Determine whether the answer is 'leather' or 'cloth' by executing Python expression, based on the number from Step4.
# Step6, Visualize results.
# Initial generated program:
# BOX0=LOC(image=IMAGE,object='woman')
# IMAGE0=CROP(image=IMAGE,box=BOX0)
# BOX1=LOC(image=IMAGE0,object='bag')
# ANSWER0=COUNT(box=BOX1)
# ANSWER1=EVAL(expr="'leather' if {{ANSWER0}} > 0 else 'cloth'")
# FINAL_RESULT=RESULT(var=ANSWER1)
# Error reason:
# The old plan changes a material question into count-based fixed-choice logic, and it ignores the important constraint 'on the left'.
# Revised subquestion:
# Step1, Locate the woman on the left, and obtain bounding boxes of the woman.
# Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
# Step3, Ask what material the bag is in the cropped image region from Step2.
# Step4, Visualize results.
# Revised program:
# BOX0=LOC(image=IMAGE,object='woman on the left')
# IMAGE0=CROP(image=IMAGE,box=BOX0)
# ANSWER0=VQA(image=IMAGE0,question='What material is the bag?')
# FINAL_RESULT=RESULT(var=ANSWER0)
#
# Finally, based on the function list and revision examples, given the following question, initial planned sub-steps, initial generated program, error reason, and revised subquestions, you need to generate the revised program that uses functions to solve the revised subquestions and answer the original question correctly. Each line in the program should correspond to the revised subquestions.
#
# Rules:
# 1. Preserve the original question type.
# 2. Use the error reason as the main correction signal.
# 3. Follow the revised subquestions as the main plan.
# 4. Do not repeat the old wrong logic.
# 5. For open-ended object or attribute questions, prefer VQA on the correct image region.
# 6. For yes/no questions about attributes or relations, prefer VQA instead of fake COUNT + EVAL.
# 7. Use COUNT + EVAL only when the question is truly about counting or existence.
# 8. Output only program lines.
#
# Question: {question}
# Initial planned sub-steps:
# {init_subq}
# Initial generated program:
# {init_prog}
# Error reason:
# {error_reason}
# Revised subquestion:
# {regenerated_subquestion}
# Revised program:
#     """


        try:
            # 必须包装成列表！
            program_prompt_list = [regenerate_prog_prompt]

            program_response = llama_generator.text_completion(
                program_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            # ==============================这里开始改=======================================
            # 获取结果
            result = program_response[0]
            print("program_response的值：", program_response)
            regenerated_program = result['generation']
            regenerated_program = self.extract_program(regenerated_program)

        except Exception as e:
            print("发生了错误，具体错误是:", e)
        print("测试 regenerate_prog_prompt 重新生成的程序是 ：\n", regenerated_program)
        return regenerated_subquestion, regenerated_program

    def generate_type_of_error(self,designed_prompt):

        designed_prompt = designed_prompt

        print('------------------检查错误类型的提示词开始------------------')
        print(designed_prompt)  # 调试时可以注释掉，避免刷屏
        print('------------------检查错误类型的提示词结束------------------')

        subq_ppl = {}
        prog_ppl = {}
        subquestion = ''
        try:
            # 必须包装成列表！
            subquestion_prompt_list = [designed_prompt]

            response = llama_generator.text_completion(
                subquestion_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,  # 开启 PPL 计算
            )

            # ==============================这里开始改=======================================
            # 获取结果
            result = response[0]
            print("测试prompt回复的response的值：", response)
            subquestion = response[0]['generation']
            print("测试prompt回复的 response[0]['generation'] ：\n", subquestion)
            json_result = extract_json_fields(subquestion)
            print("json_result 的数据类型：\n", type(json_result))
            print("最后的纯 json 数据 ：\n", json_result)

            # 第三步：按字段提取你想要的数据
            if json_result:
                error_type = json_result.get("error_type")
                error_layer = json_result.get("error_layer")
                substep_ids = json_result.get("substep_ids")
                error_reason = json_result.get("error_reason")
                fix_strategy = json_result.get("fix_strategy")

                print("--- 提取成功 ---")
                print(f"Error Type: {error_type}")
                print(f"Error Layer: {error_layer}")
                print(f"Substeps: {substep_ids}")
                print(f"error_reason: {error_reason}")
                print(f"fix_strategy: {fix_strategy}")

        except Exception as e:
            print(f"Error generating program: {e}")
            import traceback
            traceback.print_exc()
            prog = 'some errors'
        return json_result
        # ==============================这里结束=======================================
        # return subquestion, prog, prog_ppl, subq_ppl


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