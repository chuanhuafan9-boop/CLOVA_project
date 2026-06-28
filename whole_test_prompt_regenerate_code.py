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


from my_tool.prompts.prompt_engineering import add_regerate_prog_subq_experience_with_reflection
from my_tool.my_utils.utils import ProgramGenerator, ProgramInterpreter
from prompts.prompt_engineering import measure_similarity, experience_pool_index, bert_feature_forexperience
import torch


from PIL import Image
from torch.utils.data import DataLoader
from Datasets.loaders import GQADataset

print("程序可见的GPU数量：", torch.cuda.device_count())  # 一定输出 1
print("程序使用的GPU编号：", torch.cuda.current_device()) # 一定输出 0
print("GPU名称：", torch.cuda.get_device_name(0)) # 会显示你GPU2的型号

import ruamel.yaml as yaml
import re
import json
from sentence_transformers import CrossEncoder
from tools.bert_feature import Text_Feature

# 导入必需的库
import numpy as np
# matplotlib.use("TkAgg")   # 一定要在 import pyplot 之前
# matplotlib.use("Agg")   # 一定要在 import pyplot 之前
import matplotlib.pyplot as plt
from scipy import stats


# 新增这一行，禁用FastTokenizer（彻底解决文件损坏报错）
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

#################dataset construction#################
train_dataset = GQADataset(split="train", balanced=True, data_path=LLM_config['GQA']['Dataset_path'], testing=False)
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
train_n_batches = len(train_dataset)


# 根据问题ID去读取json文件里面指定的json数据
def get_data_by_id(json_file_path, target_id):
    try:
        # 1. 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)

        # 2. 遍历列表寻找对应的 ID
        for item in data_list:
            # 提取当前题目的 ID，并强制转为字符串
            current_id = str(item.get("questionID"))

            # 3. 就是你写的这行核心逻辑！用 == 比较字符串
            if current_id == target_id:
                print(f"✅ 成功找到 ID 为 {target_id} 的数据！")
                return item  # 找到了就直接把这整条字典数据返回回去

        # 如果循环结束了还没找到，说明没有这个 ID
        print(f"❌ 未找到 ID 为 {target_id} 的数据。")
        return None

    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return None



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

similarity_data = []


example_experience_pool = [
"""
Input:
1. User request:
What brand is the white sneaker on the top shelf?

2. Planned sub-steps:
Step1, Locate the sneaker, and obtain bounding boxes of the sneaker.
Step2, Crop the image region of the sneaker based on bounding boxes obtained in Step1.
Step3, Locate the brand logo in the cropped image.
Step4, Count the number of bounding boxes.
Step5, Determine whether the answer is 'Nike' or 'Adidas' by executing Python expression.
Step6, Visualize results.

3. Generated code:
BOX0=LOC(image=IMAGE,object='sneaker')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='brand logo')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'Nike' if {ANSWER0} > 0 else 'Adidas'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, API_PARAMETER_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step4", "Step5"],
  "error_reason": "In Step 1, the original question explicitly stated 'the white sneaker on the top shelf', but the LOC tool only used 'sneaker', completely omitting the crucial color and spatial attributes. If there are multiple sneakers, the tool will target the wrong one. Furthermore, the original question is 'What brand...', which is an open-ended extraction task. In Steps 4 and 5, the plan forces this into a closed binary guess using COUNT and EVAL (assuming >0 means 'Nike'), resulting in an arbitrary hallucination that completely deviates from the user's intention.",
  "fix_strategy": "Step 1: Add the attributes 'white' and 'on the top shelf' to the LOC tool parameter. Steps 3, 4, and 5: Abandon LOC, COUNT, and EVAL for text/brand extraction. Pass the cropped sneaker directly to the VQA tool and ask 'What brand is this sneaker?'."
}
""",
"""
Input:
1. User request:
Is the cat sleeping on the rug next to the fireplace?

2. Planned sub-steps:
Step1, Locate the fireplace, and obtain bounding boxes.
Step2, Crop the region right to the fireplace since the question asks what is next to it.
Step3, Locate the cat in the cropped image.
Step4, Count the number of bounding boxes.
Step5, Use EVAL to return 'yes' if count is greater than 0, else 'no'.
Step6, Visualize results.

3. Generated code:
BOX0=LOC(image=IMAGE,object='fireplace')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='cat')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3"],
  "error_reason": "In Step 2, the plan makes a false spatial assumption by rigidly converting 'next to' into CROP_RIGHTOF. The cat could be on the left or in front of the fireplace, meaning the crop will likely miss the subject entirely. In Step 3, the LOC tool searches for 'cat' but completely omits the critical state 'sleeping on the rug'. Finding a standing cat not on a rug would still trigger a 'yes' in Step 5, violating the complex conditional requirements of the user's prompt.",
  "fix_strategy": "Do not decompose ambiguous relational constraints like 'next to' into rigid geometric crops. Maintain the original image context. Submit the uncropped image to the VQA tool and ask the full question 'Is the cat sleeping on the rug next to the fireplace?' to evaluate all constraints simultaneously."
}
""",
"""
Input:
1. User request:
How many people are waiting at the bus stop across the street?

2. Planned sub-steps:
Step1, Locate the bus stop across the street, and obtain bounding boxes.
Step2, Crop the image region of the bus stop.
Step3, Ask the VQA tool 'how many people are waiting?'.
Step4, Visualize results.

3. Generated code:
BOX0=LOC(image=IMAGE,object='bus stop across the street')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='how many people are waiting?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "TOOL_CAPABILITY_LIMIT",
  "error_layer": "EXECUTION",
  "substep_ids": ["Step3"],
  "error_reason": "In Step 3, the plan delegates an exact numerical counting task ('How many people...') to the VQA tool. VQA models are semantic feature extractors and are notoriously unreliable at performing exact mathematical tallies on multiple objects, often hallucinating a random number when the count exceeds 3 or 4. This tool choice leads to highly inaccurate statistics.",
  "fix_strategy": "For precise numerical counting, replace VQA with a detection-based pipeline. In Step 3, use the LOC tool to target 'people' within the cropped bus stop region, then add a Step 4 using the COUNT tool to calculate the exact number of resulting bounding boxes."
}
""",
"""
Input:
1. User request:
What text is written on the billboard above the highway?

2. Planned sub-steps:
Step1, Locate the highway.
Step2, Crop the region above the highway.
Step3, Locate the text in the cropped image.
Step4, Crop the text region.
Step5, Ask the VQA tool 'what is written here?'.

3. Generated code:
BOX0=LOC(image=IMAGE,object='highway')
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='text')
IMAGE1=CROP(image=IMAGE0,box=BOX1)
ANSWER0=VQA(image=IMAGE1,question='what is written here?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, OCR_CONTEXT_STRIPPING",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step3", "Step4"],
  "error_reason": "In Step 3, the LOC tool is incorrectly tasked with finding 'text'. Object detection models are designed for physical nouns, not semantic strings or OCR grouping. Furthermore, in Step 4, executing a tight crop around hallucinated 'text' bounding boxes strips away the visual background of the billboard, destroying the resolution and context required for the VQA tool to successfully read the characters in Step 5.",
  "fix_strategy": "Skip the secondary LOC and CROP tools intended to isolate text. Pass the broader spatial crop (IMAGE0, the region above the highway containing the billboard) directly to the VQA tool and ask it to read the text."
}
""",
"""
Input:
1. User request:
Is the window of the blue car rolled down?

2. Planned sub-steps:
Step1, Locate the blue car, and obtain bounding boxes.
Step2, Crop the blue car region.
Step3, Locate the rolled down window in the cropped image.
Step4, Count the bounding boxes.
Step5, Output yes if count > 0, else no.

3. Generated code:
BOX0=LOC(image=IMAGE,object='blue car')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='rolled down window')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step3", "Step4", "Step5"],
  "error_reason": "The original question requires verifying a complex physical state ('rolled down'). In Step 3, the plan mistakenly uses the LOC tool to detect this state. LOC models prioritize object existence over fine-grained state variations. If the tool detects the window but it is actually closed, the bounding box count will still be greater than 0, causing the EVAL logic in Step 5 to erroneously output 'yes'.",
  "fix_strategy": "Do not dismantle physical state verification queries with LOC and COUNT combinations. Pass the cropped image of the blue car (IMAGE0) directly to the VQA tool and ask the explicit boolean question 'Is the window of the car rolled down?'."
}
""",
"""
Input:
1. User request:
Is the red apple larger than the green apple?

2. Planned sub-steps:
Step1, Locate the red apple, and crop it.
Step2, Locate the green apple, and crop it.
Step3, Ask VQA on the first crop 'Is it larger?'.
Step4, Output result.

3. Generated code:
BOX0=LOC(image=IMAGE,object='red apple')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='green apple')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='Is it larger?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, CROSS_SPATIAL_COMPARISON_DESTRUCTION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step2", "Step3"],
  "error_reason": "The user is asking for a relative size comparison ('larger than') between two objects. By isolating the apples into two separate crops in Steps 1 and 2, the plan permanently destroys the shared baseline coordinate system and visual context. In Step 3, asking a VQA model 'Is it larger?' on a single, isolated image crop is a logical paradox, as the model has absolutely no reference object against which to compare it.",
  "fix_strategy": "Relative size comparisons must be evaluated holistically. Do not separate the objects into individual crops. Provide the single, original, uncropped image to the VQA tool and directly ask 'Is the red apple larger than the green apple?'."
}
""",
"""
Input:
1. User request:
Does the shadow of the building touch the parked bicycle?

2. Planned sub-steps:
Step1, Locate the shadow of the building.
Step2, Locate the parked bicycle.
Step3, Ask VQA if the boxes touch each other.
Step4, Visualize.

3. Generated code:
BOX0=LOC(image=IMAGE,object='shadow of the building')
BOX1=LOC(image=IMAGE,object='parked bicycle')
ANSWER0=VQA(image=IMAGE,question='Do the items in BOX0 and BOX1 touch each other?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step3"],
  "error_reason": "In Step 1, the plan attempts to use LOC to find a 'shadow'. Shadows are phenomenological lighting effects, not bounded physical objects; LOC tools will likely fail or output arbitrary boxes. In Step 3, the plan commits a severe API misuse by passing internal code variables ('BOX0' and 'BOX1') directly into the natural language prompt of the VQA tool. VQA cannot interpret these internal variable references.",
  "fix_strategy": "Do not attempt to pass code variables into VQA prompts, and do not use LOC for shadows or lighting. To evaluate geometric intersections or environmental lighting interactions, submit the unmodified image to the VQA model and ask 'Does the shadow of the building touch the parked bicycle?'."
}
""",
"""
Input:
1. User request:
Is the man wearing a hat holding a coffee cup?

2. Planned sub-steps:
Step1, Locate the man, and crop the region.
Step2, Locate the coffee cup in the cropped image.
Step3, Count the bounding boxes.
Step4, Use EVAL to return yes if count > 0.

3. Generated code:
BOX0=LOC(image=IMAGE,object='man')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='coffee cup')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step2", "Step4"],
  "error_reason": "In Step 1, the LOC tool omits the crucial identifying attribute 'wearing a hat', risking tracking the wrong person. More critically, the question requires verifying a kinematic interaction ('holding'). Steps 2 and 4 downgrade this action verification into a mere existence check for a 'coffee cup' inside the crop. A coffee cup sitting on a table in the background of the man's crop would falsely trigger a 'yes' output, breaking causal logic.",
  "fix_strategy": "Step 1: Add 'wearing a hat' to the LOC parameter. Step 2 & 4: Remove LOC, COUNT, and EVAL for action verification. Pass the cropped image of the correct man to the VQA tool and ask explicitly, 'Is he holding a coffee cup?'."
}
""",
"""
Input:
1. User request:
What is the material of the coat the woman on the right is wearing?

2. Planned sub-steps:
Step1, Locate the woman, and crop the region.
Step2, Locate the coat.
Step3, Count the boxes and evaluate.
Step4, Output 'leather' if >0 else 'wool'.

3. Generated code:
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='coat')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'leather' if {ANSWER0} > 0 else 'wool'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, API_PARAMETER_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step3", "Step4"],
  "error_reason": "In Step 1, the LOC parameter omits the essential spatial anchor 'on the right', guaranteeing failure in multi-person scenes. Furthermore, the query 'What is the material...' is an open-ended attribute extraction. The plan arbitrarily restricts the universe of materials to 'leather' or 'wool' and relies on the mere existence of a coat bounding box (COUNT > 0) to output 'leather'. This is a complete logical hallucination.",
  "fix_strategy": "Step 1: Update the LOC parameter to 'woman on the right'. Step 3 & 4: Discard COUNT and EVAL. Forward the cropped image to the VQA tool and ask the open-ended question: 'What is the material of her coat?'."
}
""",
"""
Input:
1. User request:
Is the gap between the bed and the wall large enough for a nightstand?

2. Planned sub-steps:
Step1, Locate the gap between the bed and the wall.
Step2, Crop the gap region.
Step3, Ask VQA if it is large enough for a nightstand.

3. Generated code:
BOX0=LOC(image=IMAGE,object='gap between the bed and the wall')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Is it large enough for a nightstand?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, NEGATIVE_SPACE_MATERIALIZATION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step2"],
  "error_reason": "In Step 1, the plan attempts to use the LOC tool to draw a bounding box around a 'gap'. A gap represents negative space—an absence of matter defined strictly by the proximity of surrounding objects. Object detection models (LOC) are trained exclusively on positive pixel clusters (physical matter). Instructing LOC to target empty space fundamentally breaches the tool's architecture, causing unpredictable crops that ruin Step 3.",
  "fix_strategy": "Never apply LOC tools to negative spaces, holes, or relative distances. To evaluate the properties of the space between objects, bypass LOC entirely and submit the uncropped scene directly to the VQA model."
}
""",
"""
Input:
1. User request:
Are the birds flying in a circular formation?

2. Planned sub-steps:
Step1, Locate the birds.
Step2, Count the number of boxes.
Step3, Determine yes or no using EVAL. If count > 5, answer yes.

3. Generated code:
BOX0=LOC(image=IMAGE,object='birds')
ANSWER0=COUNT(box=BOX0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, GESTALT_PATTERN_DISRUPTION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3"],
  "error_reason": "The user query explicitly asks to verify a macroscopic geometric pattern ('circular formation'). The plan mistakenly assumes that detecting a specific threshold quantity of birds (count > 5) mathematically guarantees a circular shape. This is a severe logical fallacy; 20 birds flying in a straight line or scattered randomly would trigger a false 'yes' output, completely ignoring the spatial layout intent of the question.",
  "fix_strategy": "Macroscopic patterns and formations cannot be validated by micro-level bounding box counts. Abandon COUNT and EVAL. Pass the full image directly to the VQA model, which possesses the gestalt visual comprehension required to recognize formations."
}
""",
"""
Input:
1. User request:
Is the dog looking at the tennis ball?

2. Planned sub-steps:
Step1, Locate the dog, and obtain bounding boxes.
Step2, Crop the region in front of the dog.
Step3, Locate the tennis ball in the cropped image.
Step4, Count the boxes and output yes if > 0.

3. Generated code:
BOX0=LOC(image=IMAGE,object='dog')
IMAGE0=CROP_FRONTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='tennis ball')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, GAZE_VECTOR_SEVERANCE",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3", "Step4"],
  "error_reason": "Determining 'looking at' requires tracing an invisible 3D line-of-sight vector from the subject's eyes to the target. In Step 2, rigidly applying a 2D CROP_FRONTOF operation completely severs this spatial vector. The ball might be far away or at an angle, meaning the crop will miss it. Furthermore, simply proving a tennis ball exists somewhere in front of the dog (COUNT > 0) does not prove the dog is actively focusing its gaze on it.",
  "fix_strategy": "Do not use geometric cropping to evaluate line-of-sight, pointing, or gaze direction. Pass the entire unaltered image to the VQA model, explicitly asking it to evaluate if the dog's gaze is directed at the tennis ball."
}
""",
"""
Input:
1. User request:
Can you see the reflection of the trees in the lake?

2. Planned sub-steps:
Step1, Locate the lake, and crop it.
Step2, Locate the reflection of the trees in the cropped image.
Step3, Count the boxes.
Step4, Use EVAL to return yes if count is greater than 0.

3. Generated code:
BOX0=LOC(image=IMAGE,object='lake')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='reflection of the trees')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, OPTICAL_REALITY_CONFLATION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3"],
  "error_reason": "In Step 2, the plan treats an optical illusion ('reflection') as a physical object to be detected by LOC. Object detectors are trained on solid entities possessing standard edge and textural features, not on the warped, rippled, inverted specular highlights found in water. Relying on COUNT > 0 from a tool not designed for optical phenomena will result in severe false negatives.",
  "fix_strategy": "Optical phenomena (reflections, refractions, translucency) require holistic reasoning. Remove the LOC 'reflection' and COUNT sequence. Pass the cropped lake image (or the full image) directly to the VQA tool and ask 'Can you see the reflection of the trees?'."
}
""",
"""
Input:

User request:
Are there more dogs than cats in the image?

Planned sub-steps:
Step1, Locate the dogs in the image and obtain bounding boxes.
Step2, Locate the cats in the image and obtain bounding boxes.
Step3, Compare the amount of dogs and cats directly using a Python expression. Output 'yes' if dogs are more than cats, else 'no'.
Step4, Visualize results.

Generated code:
BOX0=LOC(image=IMAGE,object='dog')
BOX1=LOC(image=IMAGE,object='cat')
ANSWER0=EVAL(expr="'yes' if {BOX0} > {BOX1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
"error_type": "CODE_GENERATION_SPEC_ERROR, TYPE_MISMATCH_IN_EVAL",
"error_layer": "CODE_GENERATION",
"substep_ids": ["Step3"],
"error_reason": "Data type mismatch and API specification violation in EVAL: The user requested a numerical comparison between two object categories. While Steps 1 and 2 correctly localize the objects into variables BOX0 and BOX1, the generated code in Step 3 attempts to perform a mathematical greater-than (>) operation directly on these box variables. According to the API specification, BOX0 and BOX1 represent spatial bounding box arrays (complex lists of coordinate dictionaries), not scalar integers. Attempting to evaluate {BOX0} > {BOX1} is mathematically invalid and will result in a fatal TypeError during Python execution.",
"fix_strategy": "Adhere strictly to the tool chain specifications and variable types. Before any mathematical comparison can occur, the spatial array variables must be explicitly converted to scalar integers. Inject the COUNT tool (e.g., ANSWER0=COUNT(box=BOX0) and ANSWER1=COUNT(box=BOX1)) prior to the EVAL step, and then evaluate the resulting integer variables ({ANSWER0} > {ANSWER1})."
}
""",
"""
Input:

User request:
What are the exact bounding box coordinates of the red car parked on the street?

Planned sub-steps:
Step1, Ask the VQA tool to find the red car and output its exact coordinates.
Step2, Visualize results.

Generated code:
ANSWER0=VQA(image=IMAGE,question='What are the exact [x1, y1, x2, y2] coordinates of the red car parked on the street?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
"error_type": "TOOL_SELECTION_ERROR",
"error_layer": "EXECUTION",
"substep_ids": ["Step1"],
"error_reason": "Architectural output mismatch: The user explicitly requested exact spatial bounding box coordinates ([x1, y1, x2, y2]). The plan mistakenly selects the VQA tool for this highly specialized geometric task. VQA models are architected as multimodal text generators designed to produce natural language semantic descriptions. They completely lack the continuous spatial coordinate regression capabilities required to output precise pixel boundaries. Forcing a VQA model to output coordinates will cause it to severely hallucinate a set of arbitrary numbers that do not align with the actual image space.",
"fix_strategy": "Select the appropriate tool designed specifically for spatial localization. Replace the VQA tool with the LOC tool (e.g., BOX0=LOC(image=IMAGE, object='red car')), which is mathematically trained to regress visual features into precise bounding box coordinate arrays."
}"""]

################Experience Pool feature and score###################
text_feature_extractor = Text_Feature()

# 全局缓存：经验池特征
example_experience_pool_feature = []


def extract_user_request_from_experience(experience_text):
    """
    从经验字符串里提取 User request，和 CLOVA 一样尽量用 question 本身做检索特征。
    你的经验格式是：

    Input:
    1. User request:
    xxx

    2. Planned sub-steps:
    ...
    """
    match = re.search(
        r'1\.\s*User request:\s*(.*?)\s*2\.\s*Planned sub-steps:',
        experience_text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return experience_text.strip()

# 初始化经验池
def init_example_experience_pool_feature(example_experience_pool):
    """
    初始化经验池特征。只在最开始调用一次。
    按 CLOVA 的思路：经验池里每条经验只算一次特征，后面缓存起来。
    """
    global example_experience_pool_feature

    example_experience_pool_feature = []

    for exp in example_experience_pool:
        exp_question = extract_user_request_from_experience(exp)

        with torch.no_grad():
            feature = bert_feature_forexperience(exp_question, text_feature_extractor)

        example_experience_pool_feature.append(feature)

    print("经验池特征初始化完成，当前经验数：", len(example_experience_pool_feature))

# 添加新的经验
def append_new_experience(example_experience_pool, new_experience):
    """
    新增一条经验时，像 CLOVA 的 experience_store_incorrect 一样：
    - 经验文本 append
    - 只给这一条新经验追加一个 feature
    """
    global example_experience_pool_feature

    example_experience_pool.append(new_experience)

    with torch.no_grad():
        feature = bert_feature_forexperience(new_experience, text_feature_extractor)

    example_experience_pool_feature.append(feature)

    print("已追加一条新经验，当前经验数：", len(example_experience_pool))


def selected_error_reason_example(number_of_example, example_experience_pool, UserRequest):
    """
    从经验池中选出最相似的若干条案例
    这里不再全量重算经验池特征，而是直接使用缓存好的 example_experience_pool_feature
    """
    global example_experience_pool_feature

    num_prompts = number_of_example

    # 防御性检查：如果经验池和特征缓存长度不一致，就重新初始化一次
    if len(example_experience_pool_feature) != len(example_experience_pool):
        print("检测到经验池和特征缓存长度不一致，重新初始化经验池特征...")
        init_example_experience_pool_feature(example_experience_pool)

    if len(example_experience_pool) <= num_prompts:
        print("新加的-当前子问题池数量小于等于要选的数量，直接返回所有子问题==============================================================\n")
        prompt_examples_selected = example_experience_pool
        correct_index = list(range(len(example_experience_pool)))
    else:
        print("新加的-当前子问题池数量大于要选的数量，开始检索最相似案例==============================================================\n")

        with torch.no_grad():
            question_feature = text_feature_extractor.forward(UserRequest)

        correct_index = measure_similarity(question_feature, example_experience_pool_feature)
        prompt_examples_selected = experience_pool_index(example_experience_pool, correct_index[:num_prompts])

        print('========================== 这个计划提取到的要使用的案例的索引 ===================================', correct_index[:num_prompts])

    return prompt_examples_selected

def create_designed_prompt(prompt_examples_selected,UserRequest,planned_sub_steps,generated_program):
    """
    生成 prompt
    """
    # 最开始的prompt
    designed_prompt0=f"""You are an expert for visual-task error attribution.

Input:
1. User request
2. Planned sub-steps generated by LLM
3. Code generated from the planned sub-steps

Your task:
Identify the single main error type.
Then output:
- error_type
- error_layer
- substep_ids
- error_reason
- fix_strategy

Use only the predefined error types below.
Do not invent new labels.
Do not generate code.
Do not repeat the input.
Output JSON only.

[Error Types]
1. PLAN_STEP_DECOMPOSITION_ERROR
- The planned sub-steps themselves cannot complete the user request.
- Even if the code follows the sub-steps, the task still cannot be solved.

2. TOOL_SELECTION_ERROR
- The chosen tool type is not suitable for the task.

3. CODE_GENERATION_SPEC_ERROR
- The generated code does not faithfully implement the planned sub-steps or violates tool-calling structure.

4. API_PARAMETER_ERROR
- The tool is roughly correct, but the API parameters are wrong.

5. TOOL_CAPABILITY_LIMIT
- The plan, tool, code, and parameters are mostly correct, but the tool fails due to limited capability.

6. LUCKY_CORRECT_BY_EVAL
- The code logic cannot reliably solve the task, but the final answer happens to be correct because of EVAL or lucky final judgment.

[Error Layer]
- TASK_PLANNING
- CODE_GENERATION
- EXECUTION
- FINAL_JUDGMENT

[Decision Order]
Step 1.
If the planned sub-steps themselves cannot solve the user request:
- error_type = PLAN_STEP_DECOMPOSITION_ERROR
- error_layer = TASK_PLANNING

Step 2.
Else if the selected tool type is unsuitable:
- error_type = TOOL_SELECTION_ERROR
- error_layer = CODE_GENERATION

Step 3.
Else if the code does not faithfully implement the sub-steps:
- error_type = CODE_GENERATION_SPEC_ERROR
- error_layer = CODE_GENERATION

Step 4.
Else if the tool is roughly correct but the parameters are wrong:
- error_type = API_PARAMETER_ERROR
- error_layer = CODE_GENERATION

Step 5.
Else if everything is mostly correct but the tool still fails:
- error_type = TOOL_CAPABILITY_LIMIT
- error_layer = EXECUTION

Step 6.
Else if the logic is wrong but the answer is luckily correct because of EVAL:
- error_type = LUCKY_CORRECT_BY_EVAL
- error_layer = FINAL_JUDGMENT

[Rules]
1. Output exactly one error_type.
2. Output exactly one error_layer.
3. substep_ids must contain step names such as ["Step2", "Step4"], or [] if unknown.
4. error_reason must clearly explain why this is the main error.
5. fix_strategy must clearly explain how to revise the plan/code/tool usage.
6. Do not output any text before or after the JSON.

[Output JSON Format]
{{
  "error_type": "",
  "error_layer": "",
  "substep_ids": [],
  "error_reason": "",
  "fix_strategy": ""
}}

[Some Examples]
{prompt_examples_selected}

[Now analyze the new case]
Input:
1. User request:
{UserRequest}

2. Planned sub-steps:
{planned_sub_steps}

3. Generated code:
{generated_program}

Output:
{{
"""

    designed_prompt1=f"""You are an expert for visual-task error attribution.

You will be given:
1. User request
2. Planned sub-steps generated by an LLM
3. Code generated from the planned sub-steps

Your task:
Identify the SINGLE main error.
Then output exactly one JSON object with:
- error_type
- error_layer
- substep_ids
- error_reason
- fix_strategy

You must use ONLY the predefined error types below.
Do not invent new labels.
Do not generate code.
Do not repeat the input.
Do not output any text before or after the JSON.

==================================================
[Output JSON Format]
==================================================
{{
  "error_type": "",
  "error_layer": "",
  "substep_ids": [],
  "error_reason": "",
  "fix_strategy": ""
}}

==================================================
[Allowed Error Types]
==================================================
1. PLAN_STEP_DECOMPOSITION_ERROR
- The planned sub-steps themselves cannot solve the user request.
- Even if the code faithfully follows the plan, the task still cannot be solved.
- This includes cases where the plan changes the original task type, restricts an open-ended question into fixed choices, or destroys essential context.

2. TOOL_SELECTION_ERROR
- The chosen tool type is unsuitable for the task.

3. CODE_GENERATION_SPEC_ERROR
- The generated code does not faithfully implement the planned sub-steps.
- The code changes the intended operation, query, or tool usage from the plan.

4. API_PARAMETER_ERROR
- The selected tool is roughly correct, but the tool parameters are wrong or incomplete.

5. TOOL_CAPABILITY_LIMIT
- The plan, tool choice, code, and parameters are mostly correct, but the tool still fails because of capability limitations.

6. LUCKY_CORRECT_BY_EVAL
- The logic cannot reliably solve the task, but the final answer happens to be correct due to lucky evaluation or accidental final judgment.

==================================================
[Allowed Error Layers]
==================================================
- TASK_PLANNING
- CODE_GENERATION
- EXECUTION
- FINAL_JUDGMENT

==================================================
[STRICT DECISION ORDER]
==================================================
You MUST follow this order:

Step 1. Check whether the planned sub-steps can solve the user request at all.
If NO:
- error_type = PLAN_STEP_DECOMPOSITION_ERROR
- error_layer = TASK_PLANNING

Step 2. Otherwise, check whether the chosen tool type is unsuitable.
If YES:
- error_type = TOOL_SELECTION_ERROR
- error_layer = CODE_GENERATION

Step 3. Otherwise, check whether the code fails to faithfully implement the plan.
If YES:
- error_type = CODE_GENERATION_SPEC_ERROR
- error_layer = CODE_GENERATION

Step 4. Otherwise, check whether the tool is correct but the API parameters are wrong.
If YES:
- error_type = API_PARAMETER_ERROR
- error_layer = CODE_GENERATION

Step 5. Otherwise, check whether everything is mostly correct but the tool fails because of limited capability.
If YES:
- error_type = TOOL_CAPABILITY_LIMIT
- error_layer = EXECUTION

Step 6. Otherwise, if the logic is wrong but the final answer is accidentally correct:
- error_type = LUCKY_CORRECT_BY_EVAL
- error_layer = FINAL_JUDGMENT

==================================================
[MANDATORY TASK-TYPE CHECK]
==================================================
Before choosing any error_type, you MUST first compare:

A. What the user is asking for
Possible user task types:
- object type / identity
- attribute
- relation
- counting
- yes/no
- text reading / OCR

B. What the plan/code finally tries to output
Possible final output types:
- object type / identity
- attribute
- relation
- counting
- yes/no
- fixed label
- text reading / OCR

If A and B do NOT match, then:
- You MUST choose PLAN_STEP_DECOMPOSITION_ERROR
- You MUST set error_layer = TASK_PLANNING
- You MUST treat this as the main error, even if there are also parameter mistakes in the code

This rule has highest priority.

==================================================
[HARD TRIGGER RULES]
==================================================
Apply these rules strictly:

Rule 1.
If the user asks "what type", "what animal", "what object", or another open-ended identity question,
but the plan/code outputs:
- a count
- yes/no
- a fixed label such as "front/behind"
then the main error MUST be PLAN_STEP_DECOMPOSITION_ERROR.

Rule 2.
If the user asks an open-ended attribute question,
but the plan restricts the answer to a small fixed set such as "leather or wool",
then the main error MUST be PLAN_STEP_DECOMPOSITION_ERROR.

Rule 3.
If the user asks about a spatial relation,
but the plan crops or splits the scene in a way that removes the shared spatial reference needed for reasoning,
then the main error MUST be PLAN_STEP_DECOMPOSITION_ERROR.

Rule 4.
Do NOT choose API_PARAMETER_ERROR if fixing the parameter would still not make the overall plan solvable.

Rule 5.
Do NOT choose CODE_GENERATION_SPEC_ERROR unless the code clearly fails to follow the plan itself.

Rule 6.
If the plan itself is already unsolvable, that is the main error, even if the code also has smaller mistakes.

==================================================
[HOW TO CHOOSE substep_ids]
==================================================
- Include only the steps most responsible for the main error.
- Use step names exactly like ["Step2", "Step4"].
- If the whole plan is fundamentally broken, you may include multiple core steps.
- Do not include steps that are not central to the main error.

==================================================
[HOW TO WRITE error_reason]
==================================================
error_reason MUST be one concise paragraph.
error_reason MUST follow this structure:

1. State what the user asks for.
2. State what the plan/code actually does.
3. State why this mismatch or flaw makes the task unsolvable.

Use this sentence pattern:
"The user asks for [X], but the plan/code instead [Y]. Therefore, it cannot solve the request because [Z]."

Requirements:
- Mention the user goal explicitly.
- Mention the actual behavior of the plan/code explicitly.
- If there is task-type mismatch, explicitly mention that the task objective is changed.
- Focus on the SINGLE main error only.

==================================================
[HOW TO WRITE fix_strategy]
==================================================
fix_strategy MUST be one concise paragraph.
fix_strategy MUST follow this structure:

1. Remove or revise the wrong steps.
2. State the correct operation/tool usage.
3. Ensure the final output directly answers the user request.

Use this sentence pattern:
"Remove [A]. Instead, [B], so that the final output directly answers [X]."

Requirements:
- Fix the main error only.
- Do not mention minor secondary issues unless needed for the main fix.
- The fix must align the final output with the user's original question.

==================================================
[IMPORTANT DISTINCTIONS]
==================================================
Choose PLAN_STEP_DECOMPOSITION_ERROR when:
- the plan changes the question type
- the plan removes essential context
- the plan turns an open-ended question into a binary/fixed-label output
- the plan asks the wrong final question
- the plan cannot possibly produce the requested answer

Choose TOOL_SELECTION_ERROR when:
- the plan depends on a tool category that is fundamentally inappropriate for the task

Choose CODE_GENERATION_SPEC_ERROR when:
- the plan is reasonable, but the code changes or breaks it

Choose API_PARAMETER_ERROR when:
- the plan and tool are fine in principle, but the specific object/query/parameter is wrong or incomplete

Choose TOOL_CAPABILITY_LIMIT when:
- the plan is sound, the code is faithful, the parameters are reasonable, but the tool itself is too weak

Choose LUCKY_CORRECT_BY_EVAL when:
- the logic is clearly unreliable, but the final answer happens to be correct by accident

==================================================
[FEW-SHOT EXAMPLE]
==================================================
{prompt_examples_selected}

==================================================
[FINAL RULES]
==================================================
1. Output exactly one JSON object.
2. Output exactly one error_type.
3. Output exactly one error_layer.
4. substep_ids must be a JSON list.
5. Use double quotes in JSON.
6. Do not output markdown.
7. Do not explain your reasoning outside the JSON.
8. Do not mention any error type not listed above.
9. If the plan itself is unsolvable, always prefer PLAN_STEP_DECOMPOSITION_ERROR over later-stage errors.

[Now analyze the new case]
Input:
1. User request:
{UserRequest}

2. Planned sub-steps:
{planned_sub_steps}

3. Generated code:
{generated_program}

Output:
{{
"""

    # 让师兄看的结果，对应的prompt
    designed_prompt=f"""You are an expert for visual-task error attribution.

Input:
1. User request
2. Planned sub-steps
3. Generated code

Your task:
Identify the SINGLE main error.

Output JSON only:
{{
  "error_type": "",
  "error_layer": "",
  "substep_ids": [],
  "error_reason": "",
  "fix_strategy": ""
}}

================================
[Error Types]
================================
1. PLAN_STEP_DECOMPOSITION_ERROR
- Plan changes the task type OR cannot produce the required answer.

2. TOOL_SELECTION_ERROR
- Wrong tool type.

3. CODE_GENERATION_SPEC_ERROR
- Code does not follow the plan.

4. API_PARAMETER_ERROR
- Wrong parameters.

5. TOOL_CAPABILITY_LIMIT
- Tool fails despite correct usage.

6. LUCKY_CORRECT_BY_EVAL
- Logic is wrong but final answer is accidentally correct.

================================
[Error Layer]
================================
- TASK_PLANNING
- CODE_GENERATION
- EXECUTION
- FINAL_JUDGMENT

================================
[CRITICAL RULE - TASK TYPE CHECK]
================================
You MUST do this FIRST:

Step A. Identify user question type:
- object type (what animal / what object)
- attribute (color, material, size)
- relation (in front of, next to)
- counting
- yes/no

Step B. Identify final output of the plan/code:
- category?
- number?
- yes/no?
- fixed label (e.g., "front"/"behind")?

Step C. Compare:

IF they are NOT the same:
→ MUST choose PLAN_STEP_DECOMPOSITION_ERROR
→ error_layer = TASK_PLANNING
→ STOP reasoning

--------------------------------
Common mismatch patterns:
- "what type" → but plan outputs count / yes-no / fixed label
- open-ended question → but plan restricts answers
- relation question → but plan destroys spatial context

================================
[Decision Order]
================================
ONLY if Step C passes:

1. Plan cannot solve task → PLAN_STEP_DECOMPOSITION_ERROR
2. Wrong tool → TOOL_SELECTION_ERROR
3. Code != plan → CODE_GENERATION_SPEC_ERROR
4. Wrong parameters → API_PARAMETER_ERROR
5. Tool limitation → TOOL_CAPABILITY_LIMIT
6. Lucky correct → LUCKY_CORRECT_BY_EVAL

================================
[Strict Rules]
================================
1. Only ONE error_type
2. Only ONE error_layer
3. substep_ids like ["Step2","Step4"]
4. error_reason MUST mention:
   - what the user wants
   - what the plan actually does
   - why they mismatch (if mismatch exists)
5. No extra text outside JSON

================================
[High-Impact Example]
================================
{prompt_examples_selected}

==================================================
[FINAL RULES]
==================================================
1. Output exactly one JSON object.
2. Output exactly one error_type.
3. Output exactly one error_layer.
4. substep_ids must be a JSON list.
5. Use double quotes in JSON.
6. Do not output markdown.
7. Do not explain your reasoning outside the JSON.
8. Do not mention any error type not listed above.
9. If the plan itself is unsolvable, always prefer PLAN_STEP_DECOMPOSITION_ERROR over later-stage errors.

[Now analyze the new case]
Input:
1. User request:
{UserRequest}

2. Planned sub-steps:
{planned_sub_steps}

3. Generated code:
{generated_program}

Output:
{{
"""


    return designed_prompt

# 1. 选择模型
# 推荐默认使用这个：英文 STS 相似度任务，效果和体量比较平衡
# 对两段文本的相似度打分
model_path = "/home/fanchuanhua/.cache/huggingface/hub/models--cross-encoder--stsb-roberta-base/snapshots/d576534b67143e2c70ee9966d7fdbf5835728d13"
# 2. 加载模型
# 第一次运行时，如果本地没有这个模型，会自动从 Hugging Face Hub 拉取并缓存
# device="cpu" 表示直接用 CPU 跑；如果你有 NVIDIA GPU，可以改成 "cuda"
# cache_folder 可以指定模型下载后的缓存目录，不写也可以
similarity_model = CrossEncoder(
    model_path,
    device="cpu",
    tokenizer_args = {"use_fast": False}
)
def score_of_silimarity(text1, text2):
    # 3. 输入两段英文文本

    # 4. Cross-Encoder 的输入必须是“文本对”
    text_pairs = [(text1, text2)]

    # 5. 预测相似度分数
    # 对于 stsb 系列模型，输出通常是 0 到 1
    scores = similarity_model.predict(text_pairs)

    # 6. 取出第一组分数
    score = float(scores[0])

    # 7. 打印结果
    print(f"Similarity score: {score:.4f}")

    # 8. 给一个简单的人工解释（这是经验性规则，不是模型自带）
    if score >= 0.80:
        print("Interpretation: highly similar")
    elif score >= 0.50:
        print("Interpretation: moderately similar")
    else:
        print("Interpretation: low similarity")
    return score

# 画出相似度得分的正态分布
def picture_of_similarity_scores(similarity_data):
    # ===================== 【替换成你自己的整数数组】 =====================
    # 示例：你的整数数组（包含0）
    similarity_data = np.array(similarity_data)
    # ====================================================================

    # 1. 剔除数组中的所有 0
    filtered_data = similarity_data[similarity_data != 0]

    # 2. 检查过滤后的数据（可选，方便你查看）
    print("原始数据长度：", len(similarity_data))
    print("剔除0后数据长度：", len(filtered_data))
    print("剔除0后的数据：", filtered_data)

    # 3. 设置绘图风格
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示
    plt.figure(figsize=(10, 6))  # 设置画布大小

    # 4. 绘制核心分布图
    # 绘制归一化直方图（密度分布）
    n, bins, patches = plt.hist(
        filtered_data,
        bins='auto',  # 自动计算最佳柱子数量
        density=True,  # 归一化为密度分布（匹配正态分布）
        alpha=0.6,  # 透明度
        color='skyblue',  # 颜色
        edgecolor='black',  # 边框
        label='数据分布直方图'
    )

    # 绘制核密度估计曲线（KDE，平滑分布）
    kde = stats.gaussian_kde(filtered_data)
    x_range = np.linspace(min(filtered_data), max(filtered_data), 1000)
    plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='核密度平滑曲线')

    # 拟合正态分布曲线（完美匹配你要的“正太分布”样式）
    mu, std = stats.norm.fit(filtered_data)  # 计算均值和标准差
    plt.plot(x_range, stats.norm.pdf(x_range, mu, std),
             'g--', linewidth=2, label=f'拟合正态分布\n均值={mu:.2f}, 标准差={std:.2f}')

    # 5. 图表美化
    plt.title('数据分布（已剔除0）', fontsize=14)
    plt.xlabel('数值', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.grid(alpha=0.3)  # 网格线
    plt.legend()  # 显示图例
    plt.tight_layout()  # 自动调整布局

    # 6. 显示图像
    plt.show()



interpreter = ProgramInterpreter(task=LLM_config['Task_type'])
def execute_program(program, init_state):
    """
    执行输入进去的程序
    program: 是程序
    init_state: 是 init_state = dict(IMAGE=image.convert('RGB'))
    return: 程序的执行结果
    """
    result = None
    try:
        result, prog_state, extra_out, real_loc, word_probability = interpreter.execute(program, init_state)
    except Exception as e:
        print("执行程序失败，具体的出错情况为：", e)
    return result





if __name__ == "__main__":

    # 这里不使用 prompt 生成器，只使用里面的llama模型
    program_generator = ProgramGenerator(None,None)

    # 根据错误原因生成的程序，成功完成任务的数量
    num_correct = 0
    #总数量
    num_total = 0

    # 图片的根目录
    image_root = "/home/fanchuanhua/project/CLOVA/CLOVA-tool/dataset_download/GQA/images"

    # 生成反思结果的 prompt 里面加入的例子的数量
    number_of_example_for_reflection_result = 2
    # 根据反思结果生成子步骤和程序的 prompt 里面加入的例子的数量
    number_of_example_for_regenerate_subq_prog = 5

    i = 0
    # 配置文件路径和问题 ID 字段名
    JSON_FILE = '/home/fanchuanhua/project/CLOVA/CLOVA-tool/benchmark_data/benchmark_data.json'  # 替换为你实际的 json 文件名
    regenerated_subquestion = ''
    regenerated_program = ''
    re_result = ''

    # 1. 读取 JSON 文件
    with open(JSON_FILE, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    num = 0
    for item in data_list:
        num_total += 1
        # 下面三行是用来控制处理请求的个数的
        # num += 1
        # if num > 4:
        #     break
        print(f"========================== request {i} =======================================")
        question_id = item.get("questionID")
        UserRequest = item.get("question")
        planned_sub_steps = item.get("initial substeps")
        generated_program = item.get("initial program")
        image_id = item.get("imageID")
        ground_truth = item.get("ground truth")

        # 根据 image_id 构建图片的地址，获取真实的到图片
        image_path = os.path.join(image_root, f"{image_id}.jpg")
        image = Image.open(image_path)
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        init_state = dict(IMAGE=image.convert('RGB'))

        # ID_KEY = question_id  # 根据你的数据，唯一标识符是 questionID

        #生成从经验池当中挑选出来的例子
        prompt_examples_selected = selected_error_reason_example(number_of_example_for_reflection_result,example_experience_pool,UserRequest)

        #生成设计的 prompt
        designed_prompt = create_designed_prompt(prompt_examples_selected,UserRequest,planned_sub_steps,generated_program)

        #我们自己的系统根据请求生成的错误诊断
        json_result = program_generator.generate_type_of_error(designed_prompt)

        #根据问题ID定位到的 benchmark 里面的错误分析
        # json_result_benchmark = get_data_by_id(JSON_FILE,ID_KEY)

        new_error_analysis = {
            "questionID": item.get("questionID"),  # 强烈建议保留 ID 用于后续对应
            "error_type": item.get("error_type"),
            "error_reason": item.get("error_reason"),
            "fix_strategy": item.get("fix_strategy")
        }
        benchmark_error_reason = new_error_analysis.get("error_reason")

        try:
            our_own_error_type = json_result.get("error_type")
            our_own_error_layer = json_result.get("error_layer")
            our_own_error_substep_ids = json_result.get("substep_ids")
            our_own_error_reason = json_result.get("error_reason")
            our_own_fix_strategy = json_result.get("fix_strategy")
            print("==================开始重新生成子步骤和程序=========================")
            regenerated_subquestion, regenerated_program = program_generator.regenerate_subq_and_prog_by_reflection(UserRequest,planned_sub_steps,generated_program,our_own_error_reason,number_of_example_for_regenerate_subq_prog)
            print("==================子步骤和程序重新生成完毕=========================")
            print("try - 重新生成的子步骤为：",regenerated_subquestion)
            print("try - 重新生成的程序为：",regenerated_program)
            re_result = execute_program(regenerated_program,init_state)
            print("try - 重新生成的程序的执行结果为：", re_result)
            print("try - ground truth：", ground_truth)
            if re_result == ground_truth:
                num_correct += 1

        except Exception as e:
            our_own_error_type = ""
            our_own_error_layer = ""
            our_own_error_substep_ids = ""
            our_own_error_reason = ""
            our_own_fix_strategy = ""
            print("出错类型：",type(e))
            print("出错内容：",e)
        experience_with_reflection_result = f"""
Input:
1. User request:
{UserRequest}

2. Planned sub-steps:
{planned_sub_steps}

3. Generated code:
{generated_program}

Output:
{{
"error_type": {our_own_error_type},
"error_layer": {our_own_error_layer},
"substep_ids": {our_own_error_substep_ids},
"error_reason": {our_own_error_reason},
"fix_strategy": {our_own_fix_strategy}
}}
"""
        experience_of_regenerate_subq_with_reflection_result = f"""
Input:
Question:{UserRequest}
Initial planned sub-steps:
{planned_sub_steps}
Initial generated program:
{generated_program}
Error reason:
{our_own_error_reason}
Revised subquestion:
{regenerated_subquestion}
"""
        experience_of_regenerate_program_with_reflection_result = f"""
Input:
Question:{UserRequest}
Initial planned sub-steps:
{planned_sub_steps}
Initial generated program:
{generated_program}
Error reason:
{our_own_error_reason}
Revised subquestion:
{regenerated_subquestion}
Revised program:
{regenerated_program}
"""

        # 向经验池里面添加经验
        add_regerate_prog_subq_experience_with_reflection(experience_of_regenerate_subq_with_reflection_result, type_of_experience= "subquestion")
        add_regerate_prog_subq_experience_with_reflection(experience_of_regenerate_program_with_reflection_result, type_of_experience= "program")
        append_new_experience(example_experience_pool,experience_with_reflection_result)
        i += 1
        # 7. 打印结果
        print(f"\nUserRequest:{UserRequest}\n")
        print(f"原来的-planned_sub_steps:{planned_sub_steps}\n")
        print(f"原来的-generated_program:{generated_program}\n")
        print(f"our_own_error_reason:{our_own_error_reason}\n")
        print(f"benchmark_error_reason:{benchmark_error_reason}\n")
        print(f"重新生成的子步骤为：{regenerated_subquestion}\n")
        print(f"重新生成的程序为：{regenerated_program}\n")
        print(f"重新生成的程序的执行结果为：{re_result}\n")
        print("ground truth：", ground_truth)
        if(our_own_error_reason != ""):
            score = score_of_silimarity(our_own_error_reason, benchmark_error_reason)
        else:
            score = 0
        similarity_data.append(score)
        print(f"Similarity score: {score:.4f}\n")
        print(f"相似度得分: {score:.4f}")
    print("生成反思结果的 prompt 里面加入的例子的数量:", number_of_example_for_reflection_result)
    print("根据反思结果生成子步骤和程序的 prompt 里面加入的例子的数量:", number_of_example_for_regenerate_subq_prog)
    print("所有请求的个数:", num_total)
    print("根据错误原因重新生成程序后，成功完成任务的个数:", num_correct)
    print("相似度得分的个数：",len(similarity_data))
    print("相似度得分列表：",similarity_data)
    picture_of_similarity_scores(similarity_data)
