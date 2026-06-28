import random
from tools.bert_feature import text_feature_extraction
from tools.bert_feature import Text_Feature
import torch
import numpy as np
import ruamel.yaml as yaml
from prompts.intermediate_result import DIVIDE_BY_STEP_FUNC, DIVIDE_BY_STEPBYSTEP_FUNC

LLM_config_path = 'configs/LLM_config.yaml'
LLM_config = yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

num_start_experience = LLM_config['LLaMA']['inference']['num_start_experience']
task_type = LLM_config['Task_type']

################Experience Pool###################
if task_type == 'gqa':
    from .gqa_experience_pool import FAILED_SUBQUESTION, FAILED_PROGRAM, CURATED_SUBQUESTION, CURATED_PROGRAMS, \
    REFLECTION_STEP, REFLECTION_INTERRUPT, INFERENCE, FAILED_SUBQUESTION_WITH_REFLECTION, \
    FAILED_PROGRAM_WITH_REFLECTION, EXPERIENCE_POOL_WITH_REFLECTION_RESULT
################Experience Pool###################


################reflection_divide_function###################
divide_by_step = DIVIDE_BY_STEP_FUNC[LLM_config['Task_type']]
divide_by_stepbystep = DIVIDE_BY_STEPBYSTEP_FUNC[LLM_config['Task_type']]
################reflection_divide_function###################


################Prompt Engineering###################
if task_type == 'gqa':
    PROG_INTOR = "You need to act as a programmer. Your goal is that, given a function set, several subquestions, you need to use functions to solve subquestions step by step. \n"
    FUNCTION_DESCRIPTION = \
        "LOC: This function locates the queried region in an image. Definition: LOC(image,object). Input arguments: image and object's name. Output arguments: bounding boxes of the object.\n" + \
        "COUND: This function counts the number of bounding boxes. Definition: COUNT(box). Input arguments: bounding boxes. Output arguments: number of the bounding boxes.\n" + \
        "CROP: This function crops an image region given co-ordinates of a bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.\n" + \
        "CROP_RIGHTOF: This function crops an image region to the right of a given bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.\n" + \
        "CROP_LEFTOF: This function crops an image region to the left of a given bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.\n" + \
        "CROP_BELOW: This function crops an image region below a given bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.\n" + \
        "CROP_ABOVE: This function crops an image region above a given bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.\n" + \
        "VQA: This function generates answers of questions based on a given image. Definition: VQA(image,question). Input arguments: image and question. Output arguments: answer.\n" + \
        "EVAL: This function executes Python expression in textual form to obtain the answer. Definition: EVAL(expr). Input arguments: Python expression in textual form. Output arguments: answer.\n" + \
        "RESULT: This function is finally used to visualize results in html. Definition: RESULT(var). Input arguments: image or text. \n"
    "Except for the above ten functions, do not use other functions, such as 'IF THEN', '+'', 'max()', et al.\n"
################Prompt Engineering###################


################Fixed Prompt###################
SUBQUESTION_INTRO = "You need to act as a planner. Your goal is that, given a question, you need to decompose the complex question into a series of subquestions that can be easily executed. First, you will be given some correct decomposition examples.\n\n"
SUBQUESTION_INCORRECT_INTRO = "\nThen, you will be given some incorrect decomposition examples and their reasons. You should learn the lessons from the incorrect examples.\n\n"
SUBQUESTION_GENERATION = '\nBased on the correct examples, you need to decompose the following question into subquestions.\n\n'

# ==========根据系统出错原因生成子步骤和程序需要拼接的 prompt 前面的角色介绍
SUBQUESTION_WITH_REFLECTION_INTRO = "You need to act as a planner-reflector. Your goal is that, given a question, the initial planned sub-steps, the initial generated program, and an error reason describing the main mistake, you need to revise the subquestions so that the revised subquestions can correctly solve the original question.\n\nFirst, you will be given some correct revision examples."
SUBQUESTION_WITH_REFLECTION_GENERATION = "Based on the correct revision examples, you need to revise the following planned subquestions according to the error reason.\nRules:\n1. Preserve the original question type.\n2. Fix the main mistake described in the error reason.\n3. Keep correct parts of the original plan when possible.\n4. Remove redundant or duplicated steps.\n5. The revised subquestions must be solvable by the available functions:\nLOC(image,object)\nCOUNT(box)\nCROP(image,box)\nCROP_RIGHTOF(image,box)\nCROP_LEFTOF(image,box)\nCROP_BELOW(image,box)\nCROP_ABOVE(image,box)\nCROP_FRONTOF(image,box)\nCROP_BEHIND(image,box)\nVQA(image,question)\nEVAL(expr)\nRESULT(var)\n6. Output only revised subquestions.\n7. Every line must begin with Step1, Step2, Step3, ...\n8. The final line must be Visualize results."

PROGRAM_WITH_REFLECTION_INTRO = """
You need to act as a programmer-reflector. Your goal is that, given a function set, a question, the initial planned sub-steps, the initial generated program, an error reason describing the main mistake, and the revised subquestions, you need to generate a revised program that correctly solves the original question.

Available functions are as follows.
LOC: This function locates the queried region in an image. Definition: LOC(image,object). Input arguments: image and object's name. Output arguments: bounding boxes of the object.
COUNT: This function counts the number of bounding boxes. Definition: COUNT(box). Input arguments: bounding boxes. Output arguments: number of the bounding boxes.
CROP: This function crops an image region given co-ordinates of a bounding box. Definition: CROP(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
CROP_RIGHTOF: This function crops an image region to the right of a given bounding box. Definition: CROP_RIGHTOF(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
CROP_LEFTOF: This function crops an image region to the left of a given bounding box. Definition: CROP_LEFTOF(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
CROP_BELOW: This function crops an image region below a given bounding box. Definition: CROP_BELOW(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
CROP_ABOVE: This function crops an image region above a given bounding box. Definition: CROP_ABOVE(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
CROP_FRONTOF: This function crops an image region in front of a given bounding box. Definition: CROP_FRONTOF(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
CROP_BEHIND: This function crops an image region behind a given bounding box. Definition: CROP_BEHIND(image,box). Input arguments: image and bounding boxes. Output arguments: a cropped image.
VQA: This function generates answers of questions based on a given image. Definition: VQA(image,question). Input arguments: image and question. Output arguments: answer.
EVAL: This function executes Python expression in textual form to obtain the answer. Definition: EVAL(expr). Input arguments: Python expression in textual form. Output arguments: answer.
RESULT: This function is finally used to visualize results in html. Definition: RESULT(var). Input arguments: image or text.

Then, you will be given some correct revision examples.
"""
PROGRAM_WITH_REFLECTION_GENERATION = """
Finally, based on the function list and revision examples, given the following question, initial planned sub-steps, initial generated program, error reason, and revised subquestions, you need to generate the revised program that uses functions to solve the revised subquestions and answer the original question correctly. Each line in the program should correspond to the revised subquestions.

Rules:
1. Preserve the original question type.
2. Use the error reason as the main correction signal.
3. Follow the revised subquestions as the main plan.
4. Do not repeat the old wrong logic.
5. For open-ended object or attribute questions, prefer VQA on the correct image region.
6. For yes/no questions about attributes or relations, prefer VQA instead of fake COUNT + EVAL.
7. Use COUNT + EVAL only when the question is truly about counting or existence.
8. Output only program lines.

"""
# ==========根据系统出错原因生成子步骤和程序需要拼接的 prompt 前面的角色介绍

FUNCTION_INTRO = "\nAvailable functions are as follows.\n"
PROGRAM_CORRECT_INTRO = "\nThen, you will be given are some correct examples.\n\n"
QUESTION_INTRO = "\nFinally, based on the function list and examples, given the following subquestions, you need to generate the program that uses functions to solve the subquestions. Each line in the program corresponds to a step in the subquestions.\n\n"
PROGRAM_INCORRECT_INTRO = "\nNext, you will be given some incorrect programs and their reasons. You should learn the lessons from the incorrect examples.\n\n"

REFLECTION_INTRO_INTERRUPT = "\nYou are a debugger. You will be given a failed case including a question, the corresponding program. The program has bug, and cannot be run. You need to check which step of the program has the bug. The common error is using undefined functions and variables.\n"
REFLECTION_CORRECTEXAMPLE_INTRO2 = "\nFirst, some correct examples are provided for reference.\n\n"
REFLECTION_INTRO2_EXAMPLE = '\nFollowing are some debugging examples.\n\n'
REFLECTION_INCORRECT_INTERRUPT = '\nNext, the failed case is as follows. '
REFLECTION_OURS_INTERRUPT = 'Analyze which step of the above program caused the bug. The common error is using undefined functions and variables.\n\n'


REFLECTION_INTRO2 = "\nYou are a debugger. Your goal is that, given a failed case including a question, human feedback, our wrong answer, our decomposed subquestion, our programs, and our intermediate results obtained in each step of the program, you need to check the root cause of the wrong answer.\n"
REFLECTION_INTRO2_EXAMPLE = '\nFollowing are some debugging examples.\n\n'
REFLECTION_INCORRECT_CASE2 = '\nThe failed case needs to be debugged is as follows. \n\n'
REFLECTION_OURS2 = 'Based on the expected answer and the intermediate results we actually got at each step of the program, you need to analyze the cause of the wrong answer. \
Errors may exist in decomposed subquestions, in program, or in functions called by the program (sometimes, the functions cannot complete the intension of the program). You should locate the error, analyze it, and try to say the correct one.\
\n'

# 加了点东西111111111111111111111111111111111
REFLECTION_INTRO_STEP = """You are a debugger. Your goal is to analyze a failed step and determine if it is correct.

--------------------------------------------------
### Example 1 (Correct Case)
Question: Find the red apple.
Step: Step1
Program: BOX0=LOC(image=IMAGE, object='red apple')
Result of BOX0: [[10, 10, 50, 50]]

Is this Step1 correct?
### Your Judgment:
Yes

--------------------------------------------------
### Example 2 (Incorrect Case)
Question: What is on the white wall?
Step: Step1
Program: BOX0=LOC(image=IMAGE, object='white wall')
Result of BOX0: is empty

Is this Step1 correct?
### Your Judgment:
No, Error Location: called function in the program, Reason: The LOC tool failed to detect the white wall even though it exists in the image description.

--------------------------------------------------
### Current Case (Please Analyze This One)

"""

REFLECTION_QUESTION = '\nThe failed case is as follows: \n\n'
REFLECTION_CHECKED_STEP = ' have been checked: \n\n'
REFLECTION_UNCHECKED_STEP = '\nNow you need to check '
REFLECTION_OURS_STEP = '\nIs this Step'
REFLECTION_OURS_STEP_REFLECTIONDETAIL = 'Firstly, based on subquestions in previous steps, analyze whether this subquestions is the appropriate for solving the original question. \
If the subquestion is correct, check if the program corresponds to the subquesion. \
If the program is correct, based on the human feedback and obtained intermediate results in this step, you need to check whether the called function does not complete the intension of the program. \n'

# 1-使用gemini给我改的提示词
REFLECTION_OURS_REASON = ("""If correct, output "Yes".
If incorrect, output "No," followed by location and reason.

### Your Judgment:
""")

INFERENCE_INTRO = "\nYou are an INFERENCE maker. Your goal is that, given a failed case including a question, human feedback, our wrong answer, our decomposed subquestion, our programs, and the analysis about the wrong step, you need to infer the desirable intermediate results of the wrong step. \
In previous analysis, we have located the wrong step, and now you need to infer the desirable intermediate results of the wrong step.\n"
INFERENCE_EXAMPLE = '\nFollowing are some inferring examples.\n\n'
INFERENCE_INCORRECT_CASE = '\nThe failed case needs to be corrected is as follows. \n\n'
INFERENCE_OURS = 'Now, you will be given the failed case that needs to infer, as follows. Based on the expected answer, the intermediate results we got at each step of the program, and the analysis of the wrong step. \
You need to INFERENCE the desirable intermediate results of the wrong VQA step.\
\n'

REACT_INTRO = "A failed attempt to address this question and its failure reason are as follows.\n\n"
REACT_GENERATE_SUBQUESTION = "\nBased on the analysis of error in the failed attempt, regenerate subquestions of this question.\n\n"
REACT_GENERATE_PROGRAM = "\nBased on the analysis of error in the failed attempt, regenerate program.\n\n"
################Fixed Prompt###################

################Experience Pool feature and score###################
text_feature_extractor = Text_Feature()


def bert_feature_forexperience(experience, text_feature_extractor):
    experience_feature = text_feature_extractor.forward(experience)

    return experience_feature


def bert_feature_forpool(experience_pool, text_feature_extractor):
    dim = 768
    num_experience = len(experience_pool)
    feature = []

    for i in range(num_experience):
        experience = experience_pool[i]
        # clova 原来的代码
        # question_index = experience.find('Question:')
        # 为了匹配我的test_prompt的代码
        question_index = experience.find('User request:')
        subquestion_index = experience.find('Subquestion:')

        experience_feature = bert_feature_forexperience(experience[question_index + 9:subquestion_index],
                                                        text_feature_extractor)
        feature.append(experience_feature)

    return feature


def Euclidean_distance(feature1, feature2):
    f1_n = feature1.shape[0]
    f2_n = feature2.shape[0]

    f1_p2 = torch.sum(feature1 * feature1, dim=1)
    f2_p2 = torch.sum(feature2 * feature2, dim=1)
    f1f2 = torch.mm(feature1, torch.transpose(feature2, 0, 1))

    dis = f1_p2.expand(f1_n, f2_n) + f2_p2.expand(f1_n, f2_n) - 2 * f1f2

    return dis


def measure_similarity(feature1, feature2_list):
    feature2 = torch.cat(feature2_list, dim=0)
    sim = torch.mm(feature1, torch.transpose(feature2, 0, 1))
    resort_feature, index = torch.sort(sim[0:], descending=True)

    index = index.squeeze().cpu()
    return index


def experience_pool_index(pool, index):
    new_pool = []
    for i in range(index.shape[0]):
        new_pool.append(pool[index[i]])
    return new_pool


FAILED_SUBQUESTION_feature = bert_feature_forpool(FAILED_SUBQUESTION, text_feature_extractor)
print('FAILED_SUBQUESTION_feature', len(FAILED_SUBQUESTION_feature), FAILED_SUBQUESTION_feature[0].shape)
FAILED_PROGRAM_feature = bert_feature_forpool(FAILED_PROGRAM, text_feature_extractor)
print('FAILED_PROGRAM_feature', len(FAILED_PROGRAM_feature), FAILED_PROGRAM_feature[0].shape)

# 根据出错原因要重新生成程序和子步骤新加的,以及生成出错原因的经验池================================
FAILED_SUBQUESTION_WITH_REFLECTION_feature = bert_feature_forpool(FAILED_SUBQUESTION_WITH_REFLECTION, text_feature_extractor)
print('FAILED_SUBQUESTION_WITH_REFLECTION_feature', len(FAILED_SUBQUESTION_WITH_REFLECTION_feature), FAILED_SUBQUESTION_WITH_REFLECTION_feature[0].shape)
FAILED_PROGRAM_WITH_REFLECTION_feature = bert_feature_forpool(FAILED_PROGRAM_WITH_REFLECTION, text_feature_extractor)
print('FAILED_PROGRAM_WITH_REFLECTION_feature', len(FAILED_PROGRAM_WITH_REFLECTION_feature), FAILED_PROGRAM_WITH_REFLECTION_feature[0].shape)
EXPERIENCE_POOL_WITH_REFLECTION_RESULT_feature = bert_feature_forpool(EXPERIENCE_POOL_WITH_REFLECTION_RESULT, text_feature_extractor)
print('EXPERIENCE_POOL_WITH_REFLECTION_RESULT_feature', len(EXPERIENCE_POOL_WITH_REFLECTION_RESULT_feature), EXPERIENCE_POOL_WITH_REFLECTION_RESULT_feature[0].shape)
# 根据出错原因要重新生成程序和子步骤新加的,以及生成出错原因的经验池================================

CURATED_SUBQUESTION_feature = bert_feature_forpool(CURATED_SUBQUESTION, text_feature_extractor)
CURATED_PROGRAMS_feature = bert_feature_forpool(CURATED_PROGRAMS, text_feature_extractor)

score = {}
score['FAILED_SUBQUESTION_score'] = np.ones(len(FAILED_SUBQUESTION_feature)) * 1000
score['FAILED_PROGRAM_score'] = np.ones(len(FAILED_PROGRAM_feature)) * 1000

score['FAILED_SUBQUESTION_WITH_REFLECTION_score'] = np.ones(len(FAILED_SUBQUESTION_WITH_REFLECTION_feature)) * 1000
score['FAILED_PROGRAM_WITH_REFLECTION_score'] = np.ones(len(FAILED_PROGRAM_WITH_REFLECTION_feature)) * 1000

score['CURATED_SUBQUESTION_score'] = np.ones(len(CURATED_SUBQUESTION_feature)) * 1000
score['CURATED_PROGRAMS_score'] = np.ones(len(CURATED_PROGRAMS_feature)) * 1000


################Experience Pool feature and score###################


def create_subquestion_prompt(inputs, pre_generated, num_prompts=8, method='random', seed=42, group=0, index=False):
    # -----------correct-------------
    prompt_examples = SUBQUESTION_INTRO
    if method == 'all':
        prompt_examples_selected = CURATED_SUBQUESTION
    elif method == 'random':
        prompt_examples_selected = random.sample(CURATED_SUBQUESTION, num_prompts)
    elif method == 'retrieval':
        if len(CURATED_SUBQUESTION) <= num_prompts:
            print("新加的-当前子问题池数量小于等于要选的数量，直接返回所有子问题==============================================================\n")
            prompt_examples_selected = CURATED_SUBQUESTION
            correct_index = list(range(len(CURATED_SUBQUESTION)))
        else:
            question_feature = text_feature_extractor.forward(inputs['question'])
            correct_index = measure_similarity(question_feature, CURATED_SUBQUESTION_feature)
            prompt_examples_selected = experience_pool_index(CURATED_SUBQUESTION, correct_index[:num_prompts])
            print("新加的-下面要用到正确案例的索引==============================================================\n")
            print()
            print()
            print('here is the plan retrieval correct_index', correct_index)
            print()
            print()
            print('新加的-新加的分隔符==============================================================\n')
            print('新加的-新加的分隔符=====================================================================\n')
            print()


    else:
        raise NotImplementedError

    prompt_examples_selected = prompt_examples_selected[::-1]
    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected
    print("新加的-检索出来的正确案例=====================================================================\n",prompt_examples)



    # -----------incorrect-------------
    if len(FAILED_SUBQUESTION) > num_start_experience:
        prompt_examples = prompt_examples + SUBQUESTION_INCORRECT_INTRO
        if method == 'all':
            prompt_incorrect_examples_selected = FAILED_SUBQUESTION
        elif method == 'random':
            if len(FAILED_SUBQUESTION) <= num_prompts:
                prompt_incorrect_examples_selected = FAILED_SUBQUESTION
            else:
                # random.seed(seed)
                prompt_incorrect_examples_selected = random.sample(FAILED_SUBQUESTION, num_prompts)
        elif method == 'retrieval':
            if len(FAILED_SUBQUESTION) <= num_prompts:
                prompt_incorrect_examples_selected = FAILED_SUBQUESTION
                failed_index = list(range(len * (FAILED_SUBQUESTION)))
            else:
                failed_index = measure_similarity(question_feature, FAILED_SUBQUESTION_feature)
                prompt_incorrect_examples_selected = experience_pool_index(FAILED_SUBQUESTION,
                                                                           failed_index[:num_prompts])
        else:
            raise NotImplementedError
        prompt_incorrect_examples_selected = prompt_incorrect_examples_selected[::-1]
        prompt_incorrect_examples_selected = '\n'.join(prompt_incorrect_examples_selected)
        prompt_examples = prompt_examples + prompt_incorrect_examples_selected
        # 新加的print
        print('新加的-错误的子问题案例==============prompt_examples======================================================   \n', prompt_examples)

    else:
        failed_index = []

    prompt_examples = prompt_examples + SUBQUESTION_GENERATION
    # 新加的print
    print('新加的-子问题的正确案例==========prompt_examples===========================================================开始  \n', prompt_examples)
    print('新加的-子问题的正确案例==========prompt_examples===========================================================结束  \n')

    if index:
        return prompt_examples + "Question: {question}\nSubquestion: ".format(**inputs), correct_index[
                                                                                         :num_prompts], failed_index[
                                                                                                        :num_prompts]
    else:
        return prompt_examples + "Question: {question}\nSubquestion: ".format(**inputs)

# 自己加的 =========================================================================
# 构建根据反思结果重新生成子步骤的 prompt
def create_regerate_subquestion_prompt_with_reflection(inputs, num_prompts=8, method='retrieval'):
    """
        input 是一个字典，里面包括的键有：
            inputs["question"] = question
            inputs["init_subq"] = init_subq
            inputs["init_prog"] = init_prog
            inputs["error_reason"] = error_reason
        """
    # -----------correct-------------
    prompt_examples = SUBQUESTION_WITH_REFLECTION_INTRO
    if method == 'all':
        prompt_examples_selected = FAILED_SUBQUESTION_WITH_REFLECTION
    elif method == 'random':
        prompt_examples_selected = random.sample(FAILED_SUBQUESTION_WITH_REFLECTION, num_prompts)
    elif method == 'retrieval':
        if len(FAILED_SUBQUESTION_WITH_REFLECTION) <= num_prompts:
            print("新加的-当前子问题池数量小于等于要选的数量，直接返回所有子问题==============================================================\n")
            prompt_examples_selected = FAILED_SUBQUESTION_WITH_REFLECTION
            correct_index = list(range(len(FAILED_SUBQUESTION_WITH_REFLECTION)))
        else:
            question_feature = text_feature_extractor.forward(inputs['question'])
            correct_index = measure_similarity(question_feature, FAILED_SUBQUESTION_WITH_REFLECTION_feature)
            prompt_examples_selected = experience_pool_index(FAILED_SUBQUESTION_WITH_REFLECTION, correct_index[:num_prompts])
            print("新加的-下面要用到正确案例的索引==============================================================\n")
            print()
            print()
            print('here is the plan retrieval correct_index', correct_index)
            print()
            print()

    else:
        raise NotImplementedError

    prompt_examples_selected = prompt_examples_selected[::-1]
    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected
    print("=============根据出错原因生成子步骤检索出来的正确案例-开始==========================================\n")
    print(prompt_examples_selected)
    print("=============根据出错原因生成子步骤检索出来的正确案例-结束==========================================\n")

    prompt_examples = prompt_examples + SUBQUESTION_WITH_REFLECTION_GENERATION
    # 新加的print
    # print('新加的-子问题的正确案例==========prompt_examples===========================================================开始  \n', prompt_examples)
    # print('新加的-子问题的正确案例==========prompt_examples===========================================================结束  \n')

    question = inputs["question"]
    init_subq = inputs["init_subq"]
    init_prog = inputs["init_prog"]
    error_reason = inputs["error_reason"]


    return prompt_examples + f"Question: {question}\n\nInitial planned sub-steps:\n{init_subq}\n\nInitial generated program:\n{init_prog}\n\nError reason:\n{error_reason}\n\nRevised subquestion:"

def create_regerate_program_prompt_with_reflection(inputs, num_prompts=8, method='retrieval'):
    """
    input 是一个字典，里面包括的键有：
        inputs["question"] = question
        inputs["init_subq"] = init_subq
        inputs["init_prog"] = init_prog
        inputs["error_reason"] = error_reason
        inputs["regenerated_subquestion"] = regenerated_subquestion
    """
    # -----------correct-------------
    prompt_examples = PROGRAM_WITH_REFLECTION_INTRO
    if method == 'all':
        prompt_examples_selected = FAILED_PROGRAM_WITH_REFLECTION
    elif method == 'random':
        prompt_examples_selected = random.sample(FAILED_PROGRAM_WITH_REFLECTION, num_prompts)
    elif method == 'retrieval':
        if len(FAILED_PROGRAM_WITH_REFLECTION) <= num_prompts:
            print("新加的-当前子问题池数量小于等于要选的数量，直接返回所有子问题==============================================================\n")
            prompt_examples_selected = FAILED_PROGRAM_WITH_REFLECTION
            correct_index = list(range(len(FAILED_PROGRAM_WITH_REFLECTION)))
        else:
            question_feature = text_feature_extractor.forward(inputs['question'])
            correct_index = measure_similarity(question_feature, FAILED_PROGRAM_WITH_REFLECTION_feature)
            prompt_examples_selected = experience_pool_index(FAILED_PROGRAM_WITH_REFLECTION, correct_index[:num_prompts])
            print("新加的-下面要用到正确案例的索引==============================================================\n")
            print()
            print()
            print('here is the plan retrieval correct_index', correct_index)
            print()
            print()

    else:
        raise NotImplementedError

    prompt_examples_selected = prompt_examples_selected[::-1]
    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected
    print("=============根据出错原因生成程序检索出来的正确案例-开始==========================================\n")
    print(prompt_examples_selected)
    print("=============根据出错原因生成程序检索出来的正确案例-结束==========================================\n")

    prompt_examples = prompt_examples + PROGRAM_WITH_REFLECTION_GENERATION
    # 新加的print
    # print('新加的-子问题的正确案例==========prompt_examples===========================================================开始  \n', prompt_examples)
    # print('新加的-子问题的正确案例==========prompt_examples===========================================================结束  \n')

    question = inputs["question"]
    init_subq = inputs["init_subq"]
    init_prog = inputs["init_prog"]
    error_reason = inputs["error_reason"]
    regenerated_subquestion = inputs["regenerated_subquestion"]


    return prompt_examples + f"Question: {question}\n\nInitial planned sub-steps:\n{init_subq}\n\nInitial generated program:\n{init_prog}\n\nError reason:\n{error_reason}\n\nRevised subquestion:\n{regenerated_subquestion}\n\nRevised program:"

# 往重新生成子步骤或者程序这两个经验池里面加经验
def add_regerate_prog_subq_experience_with_reflection(experience, type_of_experience):
    """
    experience: 当前要追加的经验
    type_of_experience: 当时是追加重新生成 subquestion 的经验还是重新生成 program 的经验
    """
    if type_of_experience == "subquestion":
        FAILED_SUBQUESTION_WITH_REFLECTION.append(experience)
        with torch.no_grad():
            feature = bert_feature_forexperience(experience, text_feature_extractor)
        FAILED_SUBQUESTION_WITH_REFLECTION_feature.append(feature)

        print("根据出错原因重新生成子步骤的经验池已追加一条新经验，当前经验数：", len(FAILED_SUBQUESTION_WITH_REFLECTION))
    if type_of_experience == "program":
        FAILED_PROGRAM_WITH_REFLECTION.append(experience)
        with torch.no_grad():
            feature = bert_feature_forexperience(experience, text_feature_extractor)
        FAILED_PROGRAM_WITH_REFLECTION_feature.append(feature)
        print("根据出错原因重新生成程序的经验池已追加一条新经验，当前经验数：", len(FAILED_PROGRAM_WITH_REFLECTION))
def add_experience_of_reflection_result_with_reflection(experience):
    """
    experience: 当前要追加的经验
    type_of_experience: 当时是追加重新生成 subquestion 的经验还是重新生成 program 的经验
    """
    EXPERIENCE_POOL_WITH_REFLECTION_RESULT.append(experience)
    with torch.no_grad():
        feature = bert_feature_forexperience(experience, text_feature_extractor)
    EXPERIENCE_POOL_WITH_REFLECTION_RESULT_feature.append(feature)

    print("根据子步骤和程序生成出错原因的的经验池已追加一条新经验，当前经验数：", len(FAILED_SUBQUESTION_WITH_REFLECTION))

# 自己加的 =========================================================================



def create_program_prompt(inputs, num_prompts=8, method='all', seed=42, group=0, index=False):
    prompt_examples = prompt_examples = PROG_INTOR + FUNCTION_INTRO + FUNCTION_DESCRIPTION + PROGRAM_CORRECT_INTRO

    if method == 'all':
        prompt_examples_selected = CURATED_PROGRAMS
    elif method == 'random':
        prompt_examples_selected = random.sample(CURATED_PROGRAMS, num_prompts)
    elif method == 'retrieval':
        if len(CURATED_PROGRAMS) <= num_prompts:
            prompt_examples_selected = CURATED_PROGRAMS
            correct_index = list(range(len * (CURATED_PROGRAMS)))
        else:
            question_feature = text_feature_extractor.forward(inputs['question'])
            correct_index = measure_similarity(question_feature, CURATED_PROGRAMS_feature)
            prompt_examples_selected = experience_pool_index(CURATED_PROGRAMS, correct_index[:num_prompts])
            print("新加的-下面要用到正确案例的索引==============================================================\n")
            print()
            print()
            print('here is the progarm retrieval correct_index', correct_index)
            print()
            print()
            print('新加的-新加的分隔符==============================================================\n')
            print('新加的-新加的分隔符=====================================================================\n')
            print()

    else:
        raise NotImplementedError

    prompt_examples_selected = prompt_examples_selected[::-1]
    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected

    ##-----------incorrect-------------
    if len(FAILED_PROGRAM) > num_start_experience:
        prompt_examples = prompt_examples + PROGRAM_INCORRECT_INTRO
        if method == 'all':
            prompt_incorrect_examples_selected = FAILED_PROGRAM
        elif method == 'random':
            if len(FAILED_PROGRAM) <= num_prompts:
                prompt_incorrect_examples_selected = FAILED_PROGRAM
            else:
                prompt_incorrect_examples_selected = random.sample(FAILED_PROGRAM, num_prompts)

        elif method == 'retrieval':
            if len(FAILED_PROGRAM) <= num_prompts:
                prompt_incorrect_examples_selected = FAILED_PROGRAM
                failed_index = list(range(len * (FAILED_PROGRAM)))
            else:
                failed_index = measure_similarity(question_feature, FAILED_PROGRAM_feature)
                prompt_incorrect_examples_selected = experience_pool_index(FAILED_PROGRAM, failed_index[:num_prompts])
        else:
            raise NotImplementedError
        prompt_incorrect_examples_selected = prompt_incorrect_examples_selected[::-1]
        prompt_incorrect_examples_selected = '\n'.join(prompt_incorrect_examples_selected)
        prompt_examples = prompt_examples + prompt_incorrect_examples_selected

    else:
        failed_index = []

    prompt_examples = prompt_examples + QUESTION_INTRO

    if index:
        return prompt_examples + "Question: {question}\nSubquestion: \n{subquestion}\nProgram:".format(
            **inputs), correct_index[:num_prompts], failed_index[:num_prompts]

    else:
        return prompt_examples + "Question: {question}\nSubquestion: \n{subquestion}\nProgram:".format(**inputs)


def create_part_reflection_prompt_interrupt(inputs, num_prompts=10, method='random', seed=42, group=0):
    prompt_examples = REFLECTION_INTRO_INTERRUPT + FUNCTION_INTRO + FUNCTION_DESCRIPTION + REFLECTION_INTRO2_EXAMPLE

    if method == 'all':
        prompt_examples_selected = REFLECTION_INTERRUPT
    elif method == 'random':
        if num_prompts > len(REFLECTION_INTERRUPT):
            prompt_examples_selected = REFLECTION_INTERRUPT
        else:
            prompt_examples_selected = random.sample(REFLECTION_INTERRUPT, num_prompts)
    else:
        raise NotImplementedError

    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected
    prompt_examples = prompt_examples + REFLECTION_INCORRECT_INTERRUPT + REFLECTION_OURS_INTERRUPT

    prompt_examples = prompt_examples + "Question: {question}\nSubQuestion: \n{subquestion}\nProgram: \n{program}\nError Location:".format(
        **inputs)

    return prompt_examples


def create_part_reflection_prompt_step(inputs, prog_state, num_prompts=4, method='random', seed=42, group=0):
    step_prompt = divide_by_step(inputs['question'], inputs['human_feedback'], inputs['subquestion'], inputs['program'],
                                 inputs['intermediate_output'], prog_state)

    prompt_examples = REFLECTION_INTRO2 + REFLECTION_OURS2 + REFLECTION_INTRO2_EXAMPLE

    if method == 'all':
        prompt_examples_selected = REFLECTION_STEP
    elif method == 'random':
        if num_prompts > len(REFLECTION_STEP):
            prompt_examples_selected = REFLECTION_STEP
        else:
            prompt_examples_selected = random.sample(REFLECTION_STEP, num_prompts)
    else:
        raise NotImplementedError

    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected

    prompt_examples = prompt_examples + REFLECTION_INCORRECT_CASE2
    prompt_examples = prompt_examples + step_prompt + 'Error Location:'

    return prompt_examples


def create_part_reflection_prompt_stepbystep(inputs, prog_state, step_num, num_prompts=4, method='random', seed=42,
                                             group=0):
    d_subquestion = inputs['subquestion']
    d_subquestion = d_subquestion.split('\n')
    total_number_step = len(d_subquestion)

    prompt_examples = REFLECTION_INTRO_STEP + REFLECTION_QUESTION

    prompt_question, prompt_checked, prompt_unchecked = divide_by_stepbystep(inputs['question'],
                                                                             inputs['human_feedback'],
                                                                             inputs['subquestion'], inputs['program'],
                                                                             inputs['intermediate_output'], prog_state,
                                                                             step_num)

    if step_num == 1:
        prompt_examples = prompt_examples + prompt_question + 'It has totally ' + str(
            total_number_step) + ' steps.' + REFLECTION_UNCHECKED_STEP + 'Step1 \n\n' + prompt_unchecked
    else:
        prompt_examples = prompt_examples + prompt_question + 'It has totally ' + str(
            total_number_step) + ' steps. ' + 'Step1-Step' + str(
            step_num - 1) + REFLECTION_CHECKED_STEP + prompt_checked + REFLECTION_UNCHECKED_STEP + 'Step' + str(
            step_num) + '.\n' + prompt_unchecked

    prompt_examples = prompt_examples + REFLECTION_OURS_STEP + str(
        step_num) + ' correct? ' + REFLECTION_OURS_STEP_REFLECTIONDETAIL + REFLECTION_OURS_REASON

    return prompt_examples


def create_part_reflection_prompt_inference(inputs, prog_state, num_prompts=4, method='random', seed=42, group=0):
    step_by_step_prompt = divide_by_step(inputs['question'], inputs['human_feedback'], inputs['subquestion'],
                                         inputs['program'], inputs['intermediate_output'], prog_state)

    prompt_examples = INFERENCE_INTRO + INFERENCE_EXAMPLE

    if method == 'all':
        prompt_examples_selected = INFERENCE
    elif method == 'random':
        if len(INFERENCE) > num_prompts:
            prompt_examples_selected = random.sample(INFERENCE, num_prompts)
        else:
            prompt_examples_selected = INFERENCE
    else:
        raise NotImplementedError

    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected

    prompt_examples = prompt_examples + '\n' + INFERENCE_OURS

    prompt_examples = prompt_examples + step_by_step_prompt
    prompt_examples = prompt_examples + "Error Location: \n{errorlocation}\nReason: \n{reason}\nCorrect answer of the wrong step:".format(
        **inputs)

    return prompt_examples


def create_subquestion_react_prompt(inputs, num_prompts=8, method='random', seed=42, group=0):
    prompt_examples = SUBQUESTION_INTRO
    if method == 'all':
        prompt_examples_selected = CURATED_SUBQUESTION
    elif method == 'random':

        if num_prompts > len(CURATED_SUBQUESTION):
            prompt_examples_selected = CURATED_SUBQUESTION
        else:
            prompt_examples_selected = random.sample(CURATED_SUBQUESTION, num_prompts)

    else:
        raise NotImplementedError

    prompt_examples_selected = '\n'.join(prompt_examples_selected)
    prompt_examples = prompt_examples + prompt_examples_selected

    prompt_examples = prompt_examples + SUBQUESTION_GENERATION[:-1] + REACT_INTRO

    prompt_examples = prompt_examples + "Question: {question}\n".format(**inputs)

    prompt_examples = prompt_examples + "Subquestion: \n{subquestion}\nReason:\n{reason}\n".format(**inputs)
    prompt_examples = prompt_examples + REACT_GENERATE_SUBQUESTION

    return prompt_examples + "Question: {question}\nSubquestion: ".format(**inputs)


def create_program_react_prompt(inputs, num_prompts=8, method='all', seed=42, group=0):
    prompt_examples = prompt_examples = PROG_INTOR + FUNCTION_INTRO + FUNCTION_DESCRIPTION + PROGRAM_CORRECT_INTRO

    if method == 'all':
        prompt_examples_selected = CURATED_PROGRAMS
    elif method == 'random':

        if num_prompts > len(CURATED_PROGRAMS):
            prompt_examples_selected = CURATED_PROGRAMS
        else:
            prompt_examples_selected = random.sample(CURATED_PROGRAMS, num_prompts)
    else:
        raise NotImplementedError

    prompt_examples_selected = '\n'.join(prompt_examples_selected)

    prompt_examples = prompt_examples + prompt_examples_selected
    prompt_examples = prompt_examples + QUESTION_INTRO[:-1] + REACT_INTRO
    prompt_examples = prompt_examples + "Question: {question}\nSubquestion: \n{subquestion}\n".format(**inputs)

    prompt_examples = prompt_examples + "Program: \n{program}\nReason:\n{reason}\n".format(**inputs)
    prompt_examples = prompt_examples + REACT_GENERATE_PROGRAM

    return prompt_examples + "Question: {question}\nSubquestion: \n{newsubquestion}\nProgram:".format(**inputs)


def experience_store_incorrect(error_location, inputs):
    if 'subquestion' in error_location:
        summarized_experience = experience_summarize_incorrect(error_location, inputs)
        FAILED_SUBQUESTION.append(summarized_experience)
        feature = bert_feature_forexperience("{question}".format(**inputs), text_feature_extractor)
        FAILED_SUBQUESTION_feature.append(feature)

        FAILED_SUBQUESTION_score_list = list(score['FAILED_SUBQUESTION_score'])
        FAILED_SUBQUESTION_score_list.append(0)
        score['FAILED_SUBQUESTION_score'] = np.array(FAILED_SUBQUESTION_score_list)


    elif 'program' in error_location:
        summarized_experience = experience_summarize_incorrect(error_location, inputs)
        FAILED_PROGRAM.append(summarized_experience)
        feature = bert_feature_forexperience("{question}".format(**inputs), text_feature_extractor)
        FAILED_PROGRAM_feature.append(feature)

        FAILED_PROGRAM_score_list = list(score['FAILED_PROGRAM_score'])
        FAILED_PROGRAM_score_list.append(0)
        score['FAILED_PROGRAM_score'] = np.array(FAILED_PROGRAM_score_list)

    else:
        print('This reflection does not find the error location')


def experience_summarize_incorrect(error_location, inputs):
    if 'subquestion' in error_location:
        experience = "Question: {question}\nSubquestion: \n{subquestion}\nReason: \n{reason}\n".format(**inputs)
    elif 'program' in error_location:
        experience = "Question: {question}\nSubquestion: \n{subquestion}\nProgram: \n{program}\nReason: \n{reason}\n".format(
            **inputs)

    return experience


def experience_store_correct(inputs):
    experience_subquestion = "Question: {question}\nSubquestion: \n{subquestion}\n".format(**inputs)
    experience_program = "Question: {question}\nSubquestion: \n{subquestion}\nProgram: \n{program}\n".format(**inputs)

    CURATED_SUBQUESTION.append(experience_subquestion)
    CURATED_PROGRAMS.append(experience_program)

    experience_subquestion_feature = bert_feature_forexperience("{question}".format(**inputs), text_feature_extractor)
    experience_program_feature = bert_feature_forexperience("{question}".format(**inputs), text_feature_extractor)
    CURATED_SUBQUESTION_feature.append(experience_subquestion_feature)
    CURATED_PROGRAMS_feature.append(experience_program_feature)

    CURATED_SUBQUESTION_score_list = list(score['CURATED_SUBQUESTION_score'])
    CURATED_PROGRAMS_score_list = list(score['CURATED_PROGRAMS_score'])
    CURATED_SUBQUESTION_score_list.append(0)
    CURATED_PROGRAMS_score_list.append(0)
    score['CURATED_SUBQUESTION_score'] = np.array(CURATED_SUBQUESTION_score_list)
    score['CURATED_PROGRAMS_score'] = np.array(CURATED_PROGRAMS_score_list)


def experience_summzerize(result, selected_number):
    if result == 'correct':
        score['FAILED_SUBQUESTION_score'][selected_number[1]] = score['FAILED_SUBQUESTION_score'][
                                                                    selected_number[1]] + 1
        score['FAILED_PROGRAM_score'][selected_number[3]] = score['FAILED_PROGRAM_score'][selected_number[3]] + 1
        score['CURATED_SUBQUESTION_score'][selected_number[0]] = score['CURATED_SUBQUESTION_score'][
                                                                     selected_number[0]] + 1
        score['CURATED_PROGRAMS_score'][selected_number[2]] = score['CURATED_PROGRAMS_score'][selected_number[2]] + 1

    else:
        score['FAILED_SUBQUESTION_score'][selected_number[1]] = score['FAILED_SUBQUESTION_score'][
                                                                    selected_number[1]] - 1
        score['FAILED_PROGRAM_score'][selected_number[3]] = score['FAILED_PROGRAM_score'][selected_number[3]] - 1
        score['CURATED_SUBQUESTION_score'][selected_number[0]] = score['CURATED_SUBQUESTION_score'][
                                                                     selected_number[0]] - 1
        score['CURATED_PROGRAMS_score'][selected_number[2]] = score['CURATED_PROGRAMS_score'][selected_number[2]] - 1

    print('----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----following are scores----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----')
    print("score['FAILED_SUBQUESTION_score']", score['FAILED_SUBQUESTION_score'])
    print("score['FAILED_PROGRAM_score']", score['FAILED_PROGRAM_score'])
    print("score['CURATED_SUBQUESTION_score']", score['CURATED_SUBQUESTION_score'])
    print("score['CURATED_PROGRAMS_score']", score['CURATED_PROGRAMS_score'])
    print('----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----above are scores----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----')

    print(
        '----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----following are demonstration----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----')
    print("CURATED_SUBQUESTION", CURATED_SUBQUESTION)
    print("CURATED_PROGRAMS", CURATED_PROGRAMS)
    print('----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----above are demonstration----!!!!!!!!-----!!!!!!!!-----!!!!!!!!----')


def experience_filter():
    FAILED_SUBQUESTION_result = np.where(score['FAILED_SUBQUESTION_score'] < -2)
    FAILED_PROGRAM_result = np.where(score['FAILED_PROGRAM_score'] < -2)
    CURATED_SUBQUESTION_result = np.where(score['CURATED_SUBQUESTION_score'] < -2)
    CURATED_PROGRAMS_result = np.where(score['CURATED_PROGRAMS_score'] < -2)
    print('removed FAILED_SUBQUESTION_result', FAILED_SUBQUESTION_result)
    print('removed FAILED_PROGRAM_result', FAILED_PROGRAM_result)
    print('removed CURATED_SUBQUESTION_result', CURATED_SUBQUESTION_result)
    print('removed CURATED_PROGRAMS_result', CURATED_PROGRAMS_result)

    FAILED_SUBQUESTION_score_list = list(score['FAILED_SUBQUESTION_score'])
    FAILED_PROGRAM_score_list = list(score['FAILED_PROGRAM_score'])
    CURATED_SUBQUESTION_score_list = list(score['CURATED_SUBQUESTION_score'])
    CURATED_PROGRAMS_score_list = list(score['CURATED_PROGRAMS_score'])

    for i in range(FAILED_SUBQUESTION_result[0].shape[0]):
        FAILED_SUBQUESTION_feature.pop(FAILED_SUBQUESTION_result[0][-1 - i])
        FAILED_SUBQUESTION.pop(FAILED_SUBQUESTION_result[0][-1 - i])
        FAILED_SUBQUESTION_score_list.pop(FAILED_SUBQUESTION_result[0][-1 - i])

    for i in range(FAILED_PROGRAM_result[0].shape[0]):
        FAILED_PROGRAM_feature.pop(FAILED_PROGRAM_result[0][-1 - i])
        FAILED_PROGRAM.pop(FAILED_PROGRAM_result[0][-1 - i])
        FAILED_PROGRAM_score_list.pop(FAILED_PROGRAM_result[0][-1 - i])

    for i in range(CURATED_SUBQUESTION_result[0].shape[0]):
        CURATED_SUBQUESTION_feature.pop(CURATED_SUBQUESTION_result[0][-1 - i])
        CURATED_SUBQUESTION.pop(CURATED_SUBQUESTION_result[0][-1 - i])
        CURATED_SUBQUESTION_score_list.pop(CURATED_SUBQUESTION_result[0][-1 - i])

    for i in range(CURATED_PROGRAMS_result[0].shape[0]):
        CURATED_PROGRAMS_feature.pop(CURATED_PROGRAMS_result[0][-1 - i])
        CURATED_PROGRAMS.pop(CURATED_PROGRAMS_result[0][-1 - i])
        CURATED_PROGRAMS_score_list.pop(CURATED_PROGRAMS_result[0][-1 - i])

    print('------------following is the experience number------------')
    print('len(FAILED_SUBQUESTION_feature)', len(FAILED_SUBQUESTION_feature))
    print('len(FAILED_SUBQUESTION)', len(FAILED_SUBQUESTION))
    print('len(FAILED_SUBQUESTION_score_list)', len(FAILED_SUBQUESTION_score_list))
    print('------------------------')
    print('len(FAILED_PROGRAM_feature)', len(FAILED_PROGRAM_feature))
    print('len(FAILED_PROGRAM)', len(FAILED_PROGRAM))
    print('len(FAILED_PROGRAM_score_list)', len(FAILED_PROGRAM_score_list))
    print('------------------------')
    print('len(CURATED_SUBQUESTION_feature)', len(CURATED_SUBQUESTION_feature))
    print('len(CURATED_SUBQUESTION)', len(CURATED_SUBQUESTION))
    print('len(CURATED_SUBQUESTION_score_list)', len(CURATED_SUBQUESTION_score_list))
    print('------------------------')
    print('len(CURATED_PROGRAMS_feature)', len(CURATED_PROGRAMS_feature))
    print('len(CURATED_PROGRAMS)', len(CURATED_PROGRAMS))
    print('len(CURATED_PROGRAMS_score_list)', len(CURATED_PROGRAMS_score_list))
    print('------------------------')

    score['FAILED_SUBQUESTION_score'] = np.array(FAILED_SUBQUESTION_score_list)
    score['FAILED_PROGRAM_score'] = np.array(FAILED_PROGRAM_score_list)
    score['CURATED_SUBQUESTION_score'] = np.array(CURATED_SUBQUESTION_score_list)
    score['CURATED_PROGRAMS_score'] = np.array(CURATED_PROGRAMS_score_list)