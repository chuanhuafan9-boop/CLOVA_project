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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import sys
import pandas as pd

from my_tool.Qwen_invoke.qwen_vl_edit_judge import judge_the_result_of_imagedit

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PIL import Image
from IPython.core.display import HTML
import torch
import ruamel.yaml as yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
from framework.clova import CLOVA

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model
#################dataset construction#################
dataset_train_path=LLM_config['IMGEDIT']['dataset_train_path']
dataset_test_path=LLM_config['IMGEDIT']['dataset_test_path']
image_folder=LLM_config['IMGEDIT']['image_path']
result_save_path=LLM_config['IMGEDIT']['result_save_path']

##################存放图片的地址##################
test_result_save_path=LLM_config['IMGEDIT']['test_result_save_path']
train_result_save_path=LLM_config['IMGEDIT']['train_result_save_path']
validation_result_save_path=LLM_config['IMGEDIT']['validation_result_save_path']

if os.path.exists(result_save_path):
    print('file exists')
else:
    print('results file not exists')
    os.mkdir(result_save_path)

#################dataset construction#################



# test_count = 0

test_data_excel_list = []
#################start test before train#################
with open(dataset_test_path, 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    pbar = tqdm(lines)
    i=0
    correct_count=0
    total_count=0
    failed_prog=0

    for line in pbar:
        # 控制测试请求的数量
        # if test_count == 1:
        #     break
        # test_count += 1
        print ('\n=====================test=====================test=====================test=====================test=====================test=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

        total_count=total_count+1
        image_path,target_image_path,question=line.split(';')
        image = Image.open(image_folder+image_path)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)
        # 目标图片的地址
        target_image_path = image_folder + target_image_path

        print('=================The '+str(i)+'-th test question===============================')
        print ('------------------question------------------',question)
        print ('------------------image_path------------------', image_path) 

        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        # 定义保存图片的完整路径，方便后续同时用于保存图片和写入Excel
        save_initial_img_complete_path = test_result_save_path + question + '-test_before_train.png'
        # 保存一下原图片的地址，方便后面查看
        print("将原图片保存到：=================================================",
              test_result_save_path + question + '-test_before_train.png')
        init_state['IMAGE'].save(save_initial_img_complete_path)

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state,_,_ , test_prog_ppl, test_subq_ppl, VQA_word_probability=CLOVA_model.inference(question, init_state)
        if can_run==False:
            failed_prog=failed_prog+1
            print('program bug')

        else:
            print ('the program correctly can')
            print ('--------prog_state---------',prog_state)
            try:
                save_final_img_complete_path = result_save_path+image_path+question[:-1]+'_replace_test_before.png'
                result.save(save_final_img_complete_path)
            except:
                print ('final result is not an image.')
            print ('--------prog_state---------')
        i=i+1

        #Qwen 大模型判断结果是否正确
        result_judged_by_llm = judge_the_result_of_imagedit(save_initial_img_complete_path, save_final_img_complete_path, question)
        print("=======测试阶段由大模型判断的结果=======",result_judged_by_llm)
        if result_judged_by_llm == "true":
            correct_count += 1
        test_data_excel_list.append({
            "问题": question,
            "原图片": save_initial_img_complete_path,  # 这里存的是路径
            "标签": target_image_path,
            "测试阶段推理出来的子步骤": subq,
            "测试阶段推理出来的程序的": prog,
            "测试阶段推理结果": result,
            "推理结果的图片": save_final_img_complete_path,  # 这里存的是最后图片的
            "测试阶段由大模型判断的结果": result_judged_by_llm,
        })
    print("======= 测试阶段总共的请求数量为： =======",total_count)
    print("======= 测试阶段正确完成的请求数量为： =======",correct_count)


# ================= 循环结束后的保存代码 =================

print("正在生成带真实图片的 Excel 表格...")

# 1. 创建 DataFrame
df = pd.DataFrame(test_data_excel_list)

# 2. 定义文件名
output_file = "test_IMAGEDIT_list.xlsx"
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
    worksheet.set_column('H:H', 25)


    # ================= 插入图片的循环 =================
    print("正在逐行插入图片...")

    # 遍历每一行数据
    for idx, row in df.iterrows():
        initial_img_path = row['原图片']  # 获取路径
        target_image_path = row['标签']
        final_img_path = row['推理结果的图片']  # 获取路径
        excel_row = idx + 1  # Excel行号（从1开始，因为0是表头）

        # 1. 设置行高 (设置为 120 左右，保证图片能放得下且看清)
        worksheet.set_row(excel_row, 120)

        # 2. 插入图片
        # 你的图片是 640x640，太大了，需要缩放
        # x_scale=0.2 表示缩小到 20%，大概是 128x128 像素，刚好放入单元格
        try:
            worksheet.insert_image(excel_row, 1, initial_img_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 1, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({initial_img_path}) - {e}")

        # 插入ground truth图片
        try:
            worksheet.insert_image(excel_row, 2, target_image_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 1, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({target_image_path}) - {e}")

        # 插入结果图片
        try:
            worksheet.insert_image(excel_row, 6, final_img_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 6, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({final_img_path}) - {e}")


print(f"保存完成！请打开 {output_file} 查看效果。")
print()
print()
print()


#################start train#################
with open(dataset_train_path, 'r') as f:
    lines = f.readlines()
    n_data= len(lines)
    pbar = tqdm(lines)
    i=0

    for line in pbar:
        print ('\n=====================train=====================train=====================train=====================train=====================train=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

        total_count=total_count+1
        image_path,target_image_path,question,correct,feedback=line.split(';')
        human_feedback=feedback[:-1]
        image = Image.open(image_folder+image_path)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)

        print('=================The '+str(i)+'-th training question===============================')
        print ('------------------question------------------',question)
        print ('------------------image_path------------------', image_path)
        print ('------------------Does this question be correctly solved?------------------', int(correct)==1)
        print ('------------------human_feedback------------------', human_feedback)

        init_state = dict(
            IMAGE=image.convert('RGB')
        )


        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, _, _ , test_prog_ppl, test_subq_ppl, VQA_word_probability=CLOVA_model.inference(question, init_state)
        if int(correct)==1:
            correct_count=correct_count+1
            print ('result is correct')


        if can_run==False:
            failed_prog=failed_prog+1
            print('program bug')
        else:
            print ('the program correctly can')
            print ('------------------is the question correctedly answered?------------------', int(correct)==1)
            try:
                result.save(result_save_path+image_path+question[:-1]+'_replace_train.png')
            except:
                print ('the results are not images')


        #################reflection process#################
        inference_results=dict(can_run=can_run, correct=correct, index=index, init_state=init_state, prog_state=prog_state, question=question, subq=subq, prog=prog, human_feedback=human_feedback, answer='None')
        state, reflection_outputs = CLOVA_model.reflection(inference_results)
        print ('------------------reflection result result------------------')
        print ('state',state)
        print ('reflection_outputs',reflection_outputs)


        #################learning process#################
        if 'no_need_reflection' in state:
            learning_inputs=dict(
            question=question,
            answer='None',
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
                answer='None',
                subq=subq,
                prog=prog,
                location=reflection_outputs['location'],
                reason=reflection_outputs['reason'],
                init_state=init_state,
                prog_state=prog_state)


            else:
                learning_inputs=dict(
                question=question,
                answer='None',
                subq=reflection_outputs['new_subq'],
                prog=reflection_outputs['new_prog'],
                location=reflection_outputs['location'],
                reason=reflection_outputs['reason'],
                incorrect_subq=subq,
                incorrect_prog=prog,
                init_state=init_state,
                prog_state=reflection_outputs['new_prog_state'])

            CLOVA_model.learning(learning_inputs)


        #################report#################
        accuracy=float(correct_count/total_count)
        prog_success_ration=float(failed_prog/total_count)
        i=i+1
        pbar.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)





# validation_count = 0
validation_data_excel_list = []
#################start test after train#################
with open(dataset_test_path, 'r') as f:
    lines = f.readlines()
    n_data= len(lines)
    pbar = tqdm(lines)
    i=0

    for line in pbar:
        # 控制测试请求的数量
        # if validation_count == 1:
        #     break
        # validation_count += 1
        print ('\n=====================test=====================test=====================test=====================test=====================test=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

        total_count=total_count+1
        image_path, target_image_path, question = line.split(';')
        image = Image.open(image_folder+image_path)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)
        target_image_path = image_folder + target_image_path

        print('=================The '+str(i)+'-th test question===============================')
        print ('------------------question------------------',question)
        print ('------------------image_path------------------', image_path)

        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        # 定义保存图片的完整路径，方便后续同时用于保存图片和写入Excel
        save_initial_img_complete_path = validation_result_save_path + question + '-validation_after_train.png'
        # 保存一下原图片的地址，方便后面查看
        print("将原图片保存到：=================================================",
              validation_result_save_path + question + '-validation_after_train.png')
        init_state['IMAGE'].save(save_initial_img_complete_path)

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, _, _ , test_prog_ppl, test_subq_ppl, VQA_word_probability=CLOVA_model.inference(question, init_state)
        if can_run==False:
            failed_prog=failed_prog+1
            print ('the program has bug')

        else:
            print ('the program correctly can')
            try:
                save_final_img_complete_path = result_save_path+image_path+question[:-1]+'_replace_test_after.png'
                result.save(save_final_img_complete_path)
            except:
                print('results are not images')

        i=i+1
        #Qwen 大模型判断结果是否正确
        result_judged_by_llm = judge_the_result_of_imagedit(save_initial_img_complete_path, save_final_img_complete_path, question)
        print("=======验证阶段由大模型判断的结果=======",result_judged_by_llm)
        validation_data_excel_list.append({
            "问题": question,
            "原图片": save_initial_img_complete_path,  # 这里存的是路径
            "标签": target_image_path,
            "验证阶段推理出来的子步骤": subq,
            "验证阶段推理出来的程序的": prog,
            "验证阶段推理结果": result,
            "推理结果的图片": save_final_img_complete_path,  # 这里存的是最后图片的
            "验证阶段由大模型判断的结果": result_judged_by_llm,
        })

# ================= 循环结束后的保存代码 =================
print("正在生成带真实图片的 Excel 表格...")

# 1. 创建 DataFrame
df = pd.DataFrame(validation_data_excel_list)

# 2. 定义文件名
output_file = "validation_IMGEDIT_list.xlsx"
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
    worksheet.set_column('H:H', 25) # 由大模型判断的结果

    # ================= 插入图片的循环 =================
    print("正在逐行插入图片...")

    # 遍历每一行数据
    for idx, row in df.iterrows():
        initial_img_path = row['原图片']  # 获取路径
        final_img_path = row['推理结果的图片']  # 获取路径
        ground_truth_path = row['标签']
        excel_row = idx + 1  # Excel行号（从1开始，因为0是表头）

        # 1. 设置行高 (设置为 120 左右，保证图片能放得下且看清)
        worksheet.set_row(excel_row, 120)

        # 2. 插入图片
        # 你的图片是 640x640，太大了，需要缩放
        # x_scale=0.2 表示缩小到 20%，大概是 128x128 像素，刚好放入单元格
        try:
            worksheet.insert_image(excel_row, 1, initial_img_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 1, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({initial_img_path}) - {e}")

        # 插入ground truth图片
        try:
            worksheet.insert_image(excel_row, 2, ground_truth_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 1, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({ground_truth_path}) - {e}")

        # 插入结果图片
        try:
            worksheet.insert_image(excel_row, 6, final_img_path, {
                'x_scale': 0.2,
                'y_scale': 0.2,
                'object_position': 1  # 图片随单元格移动
            })

            # (可选) 清空原本单元格里的路径文字，只留图片
            worksheet.write_blank(excel_row, 6, None)

        except Exception as e:
            print(f"警告：第 {idx} 行图片插入失败 ({final_img_path}) - {e}")

print(f"保存完成！请打开 {output_file} 查看效果。")
print()
print()
print()