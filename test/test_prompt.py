
import os
# --- 必须加在任何其他 import 之前 --- 在pycharm里面右键调试或者运行的时候，把下面的注释打开
# 1. 告诉程序这是单机运行
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'  # 任意空闲端口

# 2. 告诉程序只有一个进程（Rank 0，World Size 1）
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# 3. 指定使用的 GPU（你之前加的那行）
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import ruamel.yaml as yaml
from llama import Llama



# LLM_config_path = 'configs/LLM_config.yaml'
LLM_config_path = '/home/fanchuanhua/project/CLOVA/CLOVA-tool/configs/LLM_config.yaml'
LLM_config = yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

ckpt_dir = LLM_config['LLaMA']['ckpt_dir_path']
tokenizer_path = LLM_config['LLaMA']['tokenizer_path']
temperature = LLM_config['LLaMA']['temperature']
top_p = LLM_config['LLaMA']['top_p']
max_seq_len = LLM_config['LLaMA']['max_seq_len']
max_gen_len = LLM_config['LLaMA']['max_gen_len']
max_batch_size = LLM_config['LLaMA']['max_batch_size']

designed_prompt = '''    你是一个严格的任务规划诊断专家。你的任务是判断：任务失败是否是由于“子步骤规划错误”导致，而不是实现阶段或执行阶段的问题。
    
    请按照下面的顺序进行思考，不要跳步：
    
    第一步：明确用户请求的最终目标。
    用一句话说明用户真正想得到的结果，包括：
    - 目标对象是什么
    - 最终需要输出什么形式的答案（例如 small/large、数字、类别名称等）
      
    第二步：检查子步骤是否始终围绕同一个目标对象。
    判断：
    - 是否有步骤改变了目标对象？
    - 是否引入了与问题无关的对象或概念？
      
    第三步：检查逻辑是否可以推出最终答案。
    判断：
    - 每一步是否为下一步提供必要输入？
    - 最终一步是否直接产生用户要求的答案？
    - 最终输出的类型是否与问题要求一致？
    - 是否缺少“从中间结果推导到最终答案”的关键步骤？
      
    第四步：做出结论。
    如果这些步骤在逻辑上无法推出用户目标，或者存在目标偏移、错误决策规则、输出类型不匹配等结构性问题，则判定为“子步骤规划错误”，并明确说明逻辑断裂发生在哪里，以及为什么这是规划层面的错误。
    如果这些步骤在逻辑上可以推出用户目标，则说明规划本身是合理的，失败更可能发生在实现或执行阶段，并说明原因。
    
    注意：
    不要假设代码有问题，除非步骤逻辑是完整的。
    不要泛泛而谈。
    必须基于“是否可以从这些步骤逻辑上推出最终答案”来判断。
    
    最后，用一段完整的话给出你的分析结论，不要分点，不要输出JSON。
    
    用户请求：
    {{UserRequest}}
    
    任务子步骤：
    {{PlannedSteps}}'''
check_part = '''
UserRequest:Is the tall clock small or large?
PlannedSteps:
Step1, Locate the tall clock, and obtain bounding boxes of the tall clock.
Step2, Crop the image region in front of the tall clock, based on bounding boxes of the tall clock. The bounding boxes are obtained in Step1.
Step3, Locate the small clock in the cropped image from Step2.
Step4, Count the number of small clock, based on bounding boxes of the small clock. The bounding boxes are obtained in Step3.
Step5, Determine if the small clock is small or large by Python expression. Answer yes if the result from Step4 are greater than 0.
Step6, Visualize results.
'''

llama_generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

def generate_type_of_error(check_part, designed_prompt):

    designed_prompt = designed_prompt

    check_part = check_part
    prompt = designed_prompt + check_part

    print('------------------检查错误类型的提示词开始------------------')
    print(prompt) # 调试时可以注释掉，避免刷屏
    print('------------------检查错误类型的提示词结束------------------')

    subq_ppl = {}
    prog_ppl = {}
    subquestion = ''
    try:
        # 必须包装成列表！
        subquestion_prompt_list = [prompt]

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


    except Exception as e:
        print(f"Error generating program: {e}")
        import traceback
        traceback.print_exc()
        prog = 'some errors'

    # ==============================这里结束=======================================
    return subquestion, prog, prog_ppl, subq_ppl

generate_type_of_error(check_part,designed_prompt)