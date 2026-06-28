import os


# --- 必须加在任何其他 import 之前 --- 在pycharm里面右键调试或者运行的时候，把下面的注释打开
# 1. 告诉程序这是单机运行
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29500'  # 任意空闲端口
#
# # 2. 告诉程序只有一个进程（Rank 0，World Size 1）
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
#
# # 3. 指定使用的 GPU（你之前加的那行）
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PIL import Image
from IPython.core.display import HTML
import torch
import ruamel.yaml as yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from framework.clova import CLOVA
from Datasets.loaders import GQADataset
from torch.utils.data import DataLoader, Subset # 需要导入 Subset
import pandas as pd  # 引入 pandas



LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)
LLM_config['Task_type']='gqa'
#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model


#################dataset construction#################
#原来训练集的地址
train_dataset = GQADataset(split="train", balanced=True, data_path=LLM_config['GQA']['Dataset_path'], testing=False)
# train_dataset = GQADataset(split="train", balanced=True, data_path="/home/fanchuanhua/project/CLOVA/"
#                                                                    "clova_project/dataset_download/GQA/"
#                                                                    "extracted_questions.json", testing=False)
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
train_n_batches = len(train_dataset)

dataset = GQADataset(split="testdev", balanced=False, data_path=LLM_config['GQA']['Dataset_path'], testing=False)
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
n_batches = len(dataset)

# # ==================== 修改开始 ====================
# # 自定义你想要跑的数据索引列表 (想要什么顺序就写什么顺序)
# custom_indices = [0, 1, 2, 3, 4]
#
# # 创建一个只包含这些索引的子数据集
# subset_dataset = Subset(train_dataset, custom_indices)
#
# # 使用子数据集创建 DataLoader
# # 注意：shuffle 必须为 False，这样才会严格按照上面列表的顺序 [44, 2, 30, 5, 10] 运行
# train_dataloader = DataLoader(subset_dataset, batch_size=1, num_workers=0, shuffle=False)
#
# # 更新总批次数，确保 tqdm 进度条显示正确 (现在长度是 5)
# train_n_batches = len(subset_dataset)
# # ==================== 修改结束 ====================

#################dataset construction#################


train_data_num=LLM_config['GQA']['train_data_num']
test_data_num=LLM_config['GQA']['test_data_num']
interval=LLM_config['GQA']['interval']

##################存放图片的地址##################
test_result_save_path=LLM_config['GQA']['test_result_save_path']
train_result_save_path=LLM_config['GQA']['train_result_save_path']
validation_result_save_path=LLM_config['GQA']['validation_result_save_path']

print("\n================= test phase begin =====================================================================================================================================================================================================================================================================================================================================================================================")
#################start testing#################
correct_count=0
total_count=0
failed_prog=0

loop = tqdm(enumerate(train_dataloader), total =train_n_batches)

# 定义您想要跳过的起始位置
start_index = 0
# 初始化一个空列表，用于存储所有数据
test_data_excel_list = []
# test phase
for i, data in loop:
    # print ('\n=====================train=====================train=====================train=====================train=====================train=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

    # 1. 跳过前 start_index 个数据
    if i < start_index:
        continue

    # 设置只训练50次（从500到550这几个问题）
    if i >= start_index + 900:
        break

    # # 原来的
    # if i > train_data_num:
    #     break

    total_count=total_count+1
    sample_id=data['sample_id'][0]
    image_id=data['image_id'][0]
    image_path=data['img'][0]
    image=Image.open(image_path)
    question=data['question'][0]
    answer=data['answer'][0]

    image.thumbnail((640,640),Image.Resampling.LANCZOS)
    init_state = dict(IMAGE=image.convert('RGB'))
    print()
    print()
    print()
    print('=================The '+str(i)+'-th testing question===============================')
    print ('------------------question------------------',question)
    print ('------------------image_id------------------',str(image_id))
    print ('------------------graound truth answer------------------', answer)
    human_feedback= f'the correct answer should be {answer}'
    print ('------------------human feedback------------------', human_feedback)
    print("------------------init_state------------------", init_state)
    print()
    print()
    print()

    # 定义保存图片的完整路径，方便后续同时用于保存图片和写入Excel
    save_img_complete_path = test_result_save_path + question[:-1] + 'test_before_train.png'
    # 保存一下原图片的地址，方便后面查看
    print("将原图片保存到：=================================================",test_result_save_path + question[:-1] + 'test_before_train.png')
    init_state['IMAGE'].save(save_img_complete_path)

    #################inference phase#################

    # word_probability 是softmax层输出的每一个单词的概率
    #test_prog_ppl 是程序的困惑值，里面有总的和每一列的
    #test_subq_ppl 是子步骤的困惑值
    can_run, subq, prog, index, result, prog_state, _, _ , test_prog_ppl, test_subq_ppl, VQA_word_probability =CLOVA_model.inference(question, init_state)

    print("完整程序为：\n",prog)
    print("程序困惑度 prog_ppl: \n",test_prog_ppl["overall_ppl"])
    print("程序每一行的困惑度 prog_lines_ppl: \n",test_prog_ppl["line_ppls"])
    print("完整子步骤为：\n", subq)
    print("子步骤困惑度 subq_ppl: \n",test_subq_ppl["overall_ppl"])
    print("子步骤每一行的困惑度 subq_lines_ppl: \n",test_subq_ppl["line_ppls"])
    print("The train "+str(i)+"-th-VQA 的 softmax 层输出的每一个单词的概率:\n",VQA_word_probability)
    # print("推理结果的困惑值 result_ppl:\n", result_ppl)


    try:
        result=str(result).lstrip('\n').rstrip('\n').lower()
        print ('新加的-生成的子步骤 ------------------subquestion--------------------------\n', subq)
        print ('新加的-生成的程序 ------------------program--------------------------\n', prog)
    except:
        print('the answer seems wrong')
    # 记录测试阶段正确完成任务的数量
    if result == answer.lower():
        correct_count = correct_count + 1
    if can_run == False:
        failed_prog = failed_prog + 1
        print('the program has bug')

    try:
        result=str(result).lstrip('\n').rstrip('\n').lower()
    except:
        print('the answer seems wrong')

    # 新加的
    else:
        print('the program correctly can')
        print('------------------prediction result------------------', result)
        print('------------------is the question correctedly answered?------------------', (result==answer))
        # accuracy = float(correct_count / total_count)
        # prog_success_ration = float(failed_prog / total_count)

        # ================= 新加代码：收集数据到列表 =================
        final_judge = "True" if result == answer else "False"
        # 无论程序运行是否成功，都尝试保存记录（如果运行失败，prog可能为空，视情况而定）
        test_data_excel_list.append({
            "问题": question,
            "图片": save_img_complete_path,  # 这里存的是路径
            "标签": answer,
            "推理出来的子步骤": subq,
            "测试阶段子步骤困惑度": test_subq_ppl,
            "推理出来的程序的": prog,
            "测试阶段程序困惑度": test_prog_ppl,
            "结果": result,
            "推理结果": final_judge, # 这里存的是 True 或 False
            # "推理结果的困惑值": result_ppl,
            "VQA 的 softmax 层输出的每一个单词的概率": VQA_word_probability
        })
        # ================= 新加代码结束 =================

        # 保留两位小数
        accuracy = round(float(correct_count / total_count), 2)
        prog_success_ration = round(float(failed_prog / total_count), 2)

        loop.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)



print("================= test phase end ===============================================================================================================================================================================================================================================================")
print()
print()
print()
print("!!!!!!!!!!!!!!!!!!!!")
print()
print()
print()
print(f"测试阶段-推理总的准确率：, {accuracy:.2f},   程序错误率：, {prog_success_ration:.2f}")
loop.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)
print()
print()
# ================= 循环结束后的保存代码 =================

print("正在生成带真实图片的 Excel 表格...")

# 1. 创建 DataFrame
df = pd.DataFrame(test_data_excel_list)

# 2. 定义文件名
output_file = "test_0~900_data_lines_PLL_excel_list.xlsx"

# 3. 使用 xlsxwriter 引擎
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # 先把文字数据写入，不包含索引
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # 获取 workbook 和 worksheet 对象
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # ================= 格式调整 =================

    # 定义自动换行的格式（用于长文本）
    text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

    # 设置列宽 (根据你的截图调整)
    worksheet.set_column('A:A', 30, text_wrap_format)  # 问题列
    worksheet.set_column('B:B', 25)  # 图片列 (宽度要够放图片)
    worksheet.set_column('C:C', 10, text_wrap_format)  # 标签
    worksheet.set_column('D:D', 50, text_wrap_format)  # 子步骤 (宽一点)
    worksheet.set_column('E:E', 25, text_wrap_format)  # 子步骤困惑度
    worksheet.set_column('F:F', 50, text_wrap_format)  # 程序 (宽一点)
    worksheet.set_column('G:G', 25, text_wrap_format)  # 程序困惑度
    worksheet.set_column('H:H', 20, text_wrap_format)  # 推理结果
    worksheet.set_column('I:I', 10, text_wrap_format)  # 结果
    worksheet.set_column('J:J', 50, text_wrap_format)  # word的概率 (宽一点)

    # ================= 插入图片的循环 =================
    print("正在逐行插入图片...")

    # 遍历每一行数据
    for idx, row in df.iterrows():
        img_path = row['图片']  # 获取路径
        excel_row = idx + 1  # Excel行号（从1开始，因为0是表头）

        # 1. 设置行高 (设置为 120 左右，保证图片能放得下且看清)
        worksheet.set_row(excel_row, 120)

        # 2. 插入图片
        # 你的图片是 640x640，太大了，需要缩放
        # x_scale=0.2 表示缩小到 20%，大概是 128x128 像素，刚好放入单元格
        try:
            worksheet.insert_image(excel_row, 1, img_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 1, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({img_path}) - {e}")

print(f"保存完成！请打开 {output_file} 查看效果。")
print()
print()
print()


print("================= train phase begin ===================================================================================================================================================================================================================================================================================================================================================================================")
#################start train#################
train_correct_count=0
train_total_count=0
train_failed_prog=0

validation_correct_count=0
validation_total_count=0
validation_failed_prog=0

# 定义您想要跳过的起始位置
start_index = 0
# 【修正：必须在这里重新定义 loop，让它重头开始】
loop = tqdm(enumerate(train_dataloader), total=train_n_batches)
# train phase
for i, data in loop:
    print ('\n=====================train=====================train=====================train=====================train=====================train=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')
    print(i)
    # 1. 跳过前 start_index 个数据
    if i < start_index:
        continue

    # 设置只训练50次（从500到550这几个问题）
    if i >= start_index + 900:
        break

    # # 原来的
    # if i > train_data_num:
    #     break

    train_total_count=train_total_count+1
    sample_id=data['sample_id'][0]
    image_id=data['image_id'][0]
    image_path=data['img'][0]
    image=Image.open(image_path)
    question=data['question'][0]
    answer=data['answer'][0]

    image.thumbnail((640,640),Image.Resampling.LANCZOS)
    init_state = dict(IMAGE=image.convert('RGB'))

    print('=================The '+str(i)+'-th training question===============================')
    print ('------------------question------------------',question)
    print ('------------------image_id------------------',str(image_id))
    print ('------------------graound truth answer------------------', answer)
    human_feedback= f'the correct answer should be {answer}'

    # 保存一下原图片的地址，方便后面查看
    print("将原图片保存到：=================================================",test_result_save_path + question[:-1] + 'test_before_train.png')
    init_state['IMAGE'].save(train_result_save_path + question[:-1] + 'training.png')



    #################inference phase#################
    can_run, subq, prog, index, result, prog_state, _, _ , train_prog_ppl, train_subq_ppl, VQA_word_probability=CLOVA_model.inference(question, init_state)

    print("完整程序为：\n",prog)
    print("程序困惑度 prog_ppl: \n",train_prog_ppl["overall_ppl"])
    print("程序每一行的困惑度 prog_lines_ppl: \n",train_prog_ppl["line_ppls"])
    print("完整子步骤为：\n", subq)
    print("子步骤困惑度 subq_ppl: \n",train_subq_ppl["overall_ppl"])
    print("子步骤每一行的困惑度 subq_lines_ppl: \n",train_subq_ppl["line_ppls"])
    print("The train "+str(i)+"-th-VQA 的 softmax 层输出的每一个单词的概率:\n",VQA_word_probability)

    try:
        result=str(result).lstrip('\n').rstrip('\n').lower()
    except:
        print('the answer seems wrong')
    if result==answer.lower():
        train_correct_count=train_correct_count+1
    if can_run==False:
        train_failed_prog=train_failed_prog+1
        print ('the program has bug')
    print ('------------------prediction result------------------', result)
    print ('------------------is the question correctedly answered?------------------', (result==answer))

    #################reflection process#################
    inference_results=dict(can_run=can_run, correct=(result==answer), index=index, init_state=init_state, prog_state=prog_state, question=question, subq=subq, prog=prog, human_feedback=human_feedback, answer=answer)
    state, reflection_outputs = CLOVA_model.reflection(inference_results)
    print ('------------------reflection result result------------------')
    print ('state',state)
    print ('reflection_outputs',reflection_outputs)

    #################learning process#################
    print("新加的-学习阶段开始==============================================================================================================================================================")
    if 'no_need_reflection' in state:
        learning_inputs=dict(
        question=question,
        answer=answer,
        subq=subq,
        prog=prog,
        location='None',
        reason='None',
        init_state=init_state,
        prog_state=prog_state)

        CLOVA_model.learning(learning_inputs)

    elif 'failed' not in state:
        if 'function' in state:
            learning_inputs=dict(
            question=question,
            answer=answer,
            subq=subq,
            prog=prog,
            location=reflection_outputs['location'],
            reason=reflection_outputs['reason'],
            init_state=init_state,
            prog_state=prog_state)

            # CLOVA_model.learning(learning_inputs)

        else:
            learning_inputs=dict(
            question=question,
            answer=answer,
            subq=reflection_outputs['new_subq'],
            prog=reflection_outputs['new_prog'],
            location=reflection_outputs['location'],
            reason=reflection_outputs['reason'],
            incorrect_subq=subq,
            incorrect_prog=prog,
            init_state=init_state,
            prog_state=reflection_outputs['new_prog_state'])

        CLOVA_model.learning(learning_inputs)
    print("新加的-学习阶段完成=========================================================================================================================================================================")

    #################report#################

    # 保留两位小数
    train_accuracy = round(float(train_correct_count / train_total_count), 2)
    train_prog_success_ration = round(float(train_failed_prog / train_total_count), 2)
    loop.set_postfix(train_accuracy=train_accuracy, prog_bug_ration=train_prog_success_ration)
    # print("训练第"+str(i)+"轮""准确率：",accuracy,"  程序错误率：",prog_success_ration)
    print("================= train phase end ===================================================================================================================================================================================================================================================================================================================================================================================")

    print()
    print()
    print()
    print()

print()
print()
print()
print()
print()
print(f"训练阶段-推理总的准确率：, {train_accuracy:.2f},   程序错误率：, {train_prog_success_ration:.2f}")
loop.set_postfix(train_accuracy=train_accuracy, train_prog_bug_ration=train_prog_success_ration)
print("================= train phase begin ===================================================================================================================================================================================================================================================================================================================================================================================")
print()
print()
print()
print()
print()




print()
print()
print()
print()
print("\n================= validation phase begin =====================================================================================================================================================================================================================================================================================================================================================================================")
#################start validating#################
correct_count=0
total_count=0
failed_prog=0

# 定义您想要跳过的起始位置
start_index = 0

# 【修正：必须在这里重新定义 loop，让它重头开始】
loop = tqdm(enumerate(train_dataloader), total=train_n_batches)
# 初始化一个空列表，用于存储所有数据
validation_data_excel_list = []
# validation phase
for i, data in loop:
    # print ('\n=====================validation=====================validation=====================train=====================validation=====================validation=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

    # 1. 跳过前 start_index 个数据
    if i < start_index:
        continue

    # 设置只训练50次（从500到550这几个问题）
    if i >= start_index + 900:
        break

    # # 原来的
    # if i > train_data_num:
    #     break

    total_count=total_count+1
    sample_id=data['sample_id'][0]
    image_id=data['image_id'][0]
    image_path=data['img'][0]
    image=Image.open(image_path)
    question=data['question'][0]
    answer=data['answer'][0]

    image.thumbnail((640,640),Image.Resampling.LANCZOS)
    init_state = dict(IMAGE=image.convert('RGB'))
    print()
    print('=================The '+str(i)+'-th validating question===============================')
    print ('------------------question------------------',question)
    print ('------------------image_id------------------',str(image_id))
    print ('------------------graound truth answer------------------', answer)
    human_feedback= f'the correct answer should be {answer}'
    print ('------------------human feedback------------------', human_feedback)
    print("------------------init_state------------------", init_state)
    print()


    # 定义保存图片的完整路径，方便后续同时用于保存图片和写入Excel
    save_img_complete_path = validation_result_save_path + question[:-1] + 'validation_after_train.png'
    # 保存一下原图片的地址，方便后面查看
    print("将原图片保存到：=================================================",test_result_save_path + question[:-1] + 'validation_after_train.png')
    init_state['IMAGE'].save(save_img_complete_path)

    #################inference phase#################
    can_run, subq, prog, index, result, prog_state, _, _ , validation_prog_ppl, validation_subq_ppl, VQA_word_probability=CLOVA_model.inference(question, init_state)

    print("完整程序为：\n",prog)
    print("程序困惑度 prog_ppl: \n",validation_prog_ppl["overall_ppl"])
    print("程序每一行的困惑度 prog_lines_ppl: \n",validation_prog_ppl["line_ppls"])
    print("完整子步骤为：\n", subq)
    print("子步骤困惑度 subq_ppl: \n",validation_subq_ppl["overall_ppl"])
    print("子步骤每一行的困惑度 subq_lines_ppl: \n",validation_subq_ppl["line_ppls"])
    print("The train "+str(i)+"-th-VQA 的 softmax 层输出的每一个单词的概率:\n",VQA_word_probability)

    # print("推理结果的困惑值 result_ppl:\n", result_ppl)



    # 记录验证阶段正确完成任务的数量
    if result==answer.lower():
        correct_count=correct_count+1
    if can_run==False:
        failed_prog=failed_prog+1
        print('the program has bug')

    try:
        result=str(result).lstrip('\n').rstrip('\n').lower()
        print ('新加的-生成的子步骤 ------------------subquestion--------------------------\n', subq)
    except:
        print('the answer seems wrong')

    # 新加的
    else:
        print('the program correctly can')
        print('------------------prediction result------------------', result)
        print('------------------is the question correctedly answered?------------------', (result==answer))
        # accuracy = float(correct_count / total_count)
        # prog_success_ration = float(failed_prog / total_count)

        # ================= 新加代码：收集数据到列表 =================
        final_judge = "True" if result == answer else "False"
        # 无论程序运行是否成功，都尝试保存记录（如果运行失败，prog可能为空，视情况而定）
        validation_data_excel_list.append({
            "问题": question,
            "图片": save_img_complete_path,  # 这里存的是路径
            "标签": answer,
            "推理出来的子步骤": subq,
            "验证阶段子步骤困惑度": validation_subq_ppl,
            "推理出来的程序的": prog,
            "验证阶段程序困惑度": validation_prog_ppl,
            "结果": result,
            "推理结果": final_judge, # 这里存的是 True 或 False
            "VQA 的 softmax 层输出的每一个单词的概率":VQA_word_probability
        })
        # ================= 新加代码结束 =================
        # 保留两位小数
        accuracy = round(float(correct_count / total_count), 2)
        prog_success_ration = round(float(failed_prog / total_count), 2)

        loop.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)
        # print("验证第" + str(i) + "轮""准确率：", accuracy, "  程序错误率：", prog_success_ration)


print("================= validation phase end ===============================================================================================================================================================================================================================================================")
print("\n\n\n\n\n")
print(f"验证阶段-推理总的准确率：, {accuracy:.2f},   程序错误率：, {prog_success_ration:.2f}")
loop.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)

# ================= 循环结束后的保存代码 =================

print("正在生成带真实图片的 Excel 表格...")

# 1. 创建 DataFrame
df = pd.DataFrame(validation_data_excel_list)

# 2. 定义文件名
output_file = "validation_0~900_data_lines_PLL_excel_list.xlsx"

# 3. 使用 xlsxwriter 引擎
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # 先把文字数据写入，不包含索引
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # 获取 workbook 和 worksheet 对象
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # ================= 格式调整 =================

    # 定义自动换行的格式（用于长文本）
    text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

    # 设置列宽 (根据你的截图调整)
    worksheet.set_column('A:A', 30, text_wrap_format)  # 问题列
    worksheet.set_column('B:B', 25)  # 图片列 (宽度要够放图片)
    worksheet.set_column('C:C', 10, text_wrap_format)  # 标签
    worksheet.set_column('D:D', 50, text_wrap_format)  # 子步骤 (宽一点)
    worksheet.set_column('E:E', 25, text_wrap_format)  # 子步骤困惑度
    worksheet.set_column('F:F', 50, text_wrap_format)  # 程序 (宽一点)
    worksheet.set_column('G:G', 25, text_wrap_format)  # 程序困惑度
    worksheet.set_column('H:H', 20, text_wrap_format)  # 推理结果
    worksheet.set_column('I:I', 10, text_wrap_format)  # 结果
    worksheet.set_column('J:J', 50, text_wrap_format)  # word的概率 (宽一点)

    # ================= 插入图片的循环 =================
    print("正在逐行插入图片...")

    # 遍历每一行数据
    for idx, row in df.iterrows():
        img_path = row['图片']  # 获取路径
        excel_row = idx + 1  # Excel行号（从1开始，因为0是表头）

        # 1. 设置行高 (设置为 120 左右，保证图片能放得下且看清)
        worksheet.set_row(excel_row, 120)

        # 2. 插入图片
        # 你的图片是 640x640，太大了，需要缩放
        # x_scale=0.2 表示缩小到 20%，大概是 128x128 像素，刚好放入单元格
        try:
            worksheet.insert_image(excel_row, 1, img_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 1, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({img_path}) - {e}")

print(f"保存完成！请打开 {output_file} 查看效果。")

