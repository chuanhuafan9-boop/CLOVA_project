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

import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PIL import Image
from IPython.core.display import HTML
import torch
import ruamel.yaml as yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from framework.clova import CLOVA
from knowtag_fl_score.ok_tag_res import judge_all_right, calculate_f1_score_single, calculate_f1_score_tag

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)


#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model

#################dataset construction#################

dataset_train_path=LLM_config['KNOWTAG']['dataset_train_path']
dataset_test_path=LLM_config['KNOWTAG']['dataset_test_path']
gt_path=LLM_config['KNOWTAG']['gt_path']
result_save_path=LLM_config['KNOWTAG']['result_save_path']

##################存放图片的地址##################
test_result_save_path=LLM_config['KNOWTAG']['test_result_save_path']
train_result_save_path=LLM_config['KNOWTAG']['train_result_save_path']
validation_result_save_path=LLM_config['KNOWTAG']['validation_result_save_path']


with open(gt_path, 'r') as file:
    data_real = file.readlines()
print ('ground truth of training data has been loaded.')

if os.path.exists(result_save_path):
    print('file exists')
else:
    print('results file not exists')
    os.mkdir(result_save_path)

before_json_train_file_name= result_save_path+'testresult_before_.json'
before_ans_file = open(before_json_train_file_name, 'a')

after_json_train_file_name= result_save_path+'testresult_after_.json'
after_ans_file = open(after_json_train_file_name, 'a')
#################dataset construction#################


train_data_num=LLM_config['KNOWTAG']['train_data_num']
test_data_num=LLM_config['KNOWTAG']['test_data_num']
interval=LLM_config['KNOWTAG']['interval']


correct_count=0
total_count=0
failed_prog=0

test_data_excel_list = []

#################start test before train#################
with open(dataset_test_path+'test.txt', 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    # pbar = tqdm(lines)
    real_count=0

    # i = 0

    for line in lines:
        # 控制处理几个任务
        # i += 1
        # if i>5:
        #     break

        print ('real_count',real_count)
        real=data_real[real_count]
        # real_json = json.loads(real.strip())
        # real_box_all = real_json['real']
        real_json = json.loads(real.strip())
        image_name=real_json['image']
        real_box_all = real_json['real']

        print ('\n================beforelearning=========================================================================================================================================================================================================================================================================================================================================================================================================================beforelearning=============================')
        image_path,instruction=line.split(';')
        instruction=instruction[:-1]
        print ('image_path:',image_path)
        print ('Instruction:',instruction)
        image = Image.open(dataset_test_path+image_path)

        image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        # 定义保存图片的完整路径，方便后续同时用于保存图片和写入Excel
        save_initial_img_complete_path = test_result_save_path + instruction + '-test_before_train.png'
        # 保存一下原图片的地址，方便后面查看
        print("将原图片保存到：=================================================",
              test_result_save_path + instruction + '-test_before_train.png')
        init_state['IMAGE'].save(save_initial_img_complete_path)

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, before_extra_out, before_real_loc, test_prog_ppl, test_subq_ppl, VQA_word_probability =CLOVA_model.inference(instruction, init_state)
        save_final_img_complete_path = ""
        if can_run==False or before_extra_out==None:
            print("===========can_run==False or before_extra_out==None============")
            failed_prog=failed_prog+1
            result=''
            before_extra_out=[]
            cur_js= {}
            cur_js['image'] = image_path
            cur_js['res'] = before_extra_out
            before_ans_file.write(json.dumps(cur_js) + '\n')
            before_ans_file.flush()
            real_count=real_count+1
        else:
            cur_js= {}
            cur_js['image'] = image_path
            if len(before_extra_out) > 0:
                for data in before_extra_out:
                    if 'category' in data.keys():
                        del data['category']
                    if 'inst_id' in data.keys():
                        del data['inst_id']
                    if 'mask' in data.keys():
                        del data['mask']
                    if 'class_score' in data.keys():
                        del data['class_score']
            cur_js['res'] = before_extra_out
            before_ans_file.write(json.dumps(cur_js) + '\n')
            before_ans_file.flush()
            real_count=real_count+1

            # 最后程序执行之后得到的图片
            if 'Image' in str(type(result)):
                save_final_img_complete_path = result_save_path+image_path+instruction+'test_before_train_result.png'
                result.save(save_final_img_complete_path)
                # 无论程序运行是否成功，都尝试保存记录（如果运行失败，prog可能为空，视情况而定）
        print("save_final_img_complete_path =:",save_final_img_complete_path)
        test_data_excel_list.append({
            "问题": instruction,
            "原图片": save_initial_img_complete_path,  # 这里存的是路径
            "标签": real_box_all,
            "测试阶段推理出来的子步骤": subq,
            "测试阶段推理出来的程序的": prog,
            "测试阶段推理结果": result,
            "推理结果的图片": save_final_img_complete_path,  # 这里存的是最后图片的
        })

# ================= 循环结束后的保存代码 =================

print("正在生成带真实图片的 Excel 表格...")

# 1. 创建 DataFrame
df = pd.DataFrame(test_data_excel_list)

# 2. 定义文件名
output_file = "test_KNOWTAG_list.xlsx"
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


    # ================= 插入图片的循环 =================
    print("正在逐行插入图片...")

    # 遍历每一行数据
    for idx, row in df.iterrows():
        initial_img_path = row['原图片']  # 获取路径
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
with open(dataset_train_path+'train.txt', 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    # pbar = tqdm(lines)
    real_count=0
    # i = 0

    for line in lines:
        # 控制处理几个任务
        # i += 1
        # if i>5:
        #     break

        print ('real_count',real_count)
        real=data_real[real_count]
        real_json = json.loads(real.strip())
        real_box_all = real_json['real']
        real_json = json.loads(real.strip())
        image_name=real_json['image']
        real_box_all = real_json['real']

        print ('\n================training=========================================================================================================================================================================================================================================================================================================================================================================================================================beforelearning=============================')
        image_path,instruction=line.split(';')
        instruction=instruction[:-1]
        print ('image_path:',image_path)
        print ('Instruction:',instruction)
        image = Image.open(dataset_train_path+image_path)

        image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, extra_out, real_loc, test_prog_ppl, test_subq_ppl, VQA_word_probability =CLOVA_model.inference(instruction, init_state)
        if can_run==False or extra_out==None:
            failed_prog=failed_prog+1
            is_all_right=False
            human_feedback='This program has bug'
            result=''
            extra_out=[]
            cur_js= {}
            cur_js['image'] = image_path
            cur_js['res'] = extra_out

        else:
            cur_js= {}
            cur_js['image'] = image_path
            if len(extra_out) > 0:
                for data in extra_out:
                    if 'category' in data.keys():
                        del data['category']
                    if 'inst_id' in data.keys():
                        del data['inst_id']
                    if 'mask' in data.keys():
                        del data['mask']
                    if 'class_score' in data.keys():
                        del data['class_score']
            cur_js['res'] = extra_out

            new_format_pred =[]
            new_format_real = []
            for data in extra_out:
                new_format_pred.append((data['box'], data['class']))
            for data in real_box_all:
                new_format_real.append((data['box'], data['class']))

            is_all_right = judge_all_right(new_format_pred, new_format_real) ### judge each image correct or incorrect
            print("signle image predictiobn: ", is_all_right)   
            f1_single = calculate_f1_score_single(new_format_pred, new_format_real) ### 计算单张图f1 score   
            print("single image f1_socre", f1_single)                             
            true_positives, false_positives, false_negatives = calculate_f1_score_tag(new_format_pred, new_format_real)

            if true_positives + false_positives==0:
                precision=0
            else:
                precision = true_positives / (true_positives + false_positives)
                
            if true_positives + false_negatives==0:
                recall=0
            else:
                recall = true_positives / (true_positives + false_negatives)

            F1_score=f1_single
            dict_gt=new_format_real.copy()
            dict_our=new_format_pred.copy()
            num_gt=len(dict_gt)
            num_our=len(dict_our)
            Our_prediction=''
            Ground_truth=f'There are {str(num_gt)} objects should be tagged, while our method tags {str(num_our)} objects. The details of desirable prediction is {str(dict_gt)}, while our prediction is {str(dict_our)}.'
            human_feedback=Our_prediction+Ground_truth

            real_count=real_count+1


        #################reflection process#################
        inference_results=dict(can_run=can_run, correct=is_all_right, index=index, init_state=init_state, prog_state=prog_state, question=instruction, subq=subq, prog=prog, human_feedback=human_feedback, answer='None')
        state, reflection_outputs = CLOVA_model.reflection(inference_results)
        print ('------------------reflection result result------------------')   
        print ('state',state)
        print ('reflection_outputs',reflection_outputs)  



        #################learning process#################
        if 'failed' in state:
            continue

        elif 'no_need_reflection' in state:
            learning_inputs=dict(
            question=instruction,
            answer='None',
            subq=subq,
            prog=prog,
            location='None',
            reason='None',
            init_state=init_state,
            prog_state=prog_state)

            CLOVA_model.learning(learning_inputs)

        else:
            if 'function' in state:
                learning_inputs=dict(
                question=instruction,
                answer='None',
                subq=subq,
                prog=prog,
                location=reflection_outputs['location'],
                reason=reflection_outputs['reason'],
                init_state=init_state,
                prog_state=prog_state)

            else:
                learning_inputs=dict(
                question=instruction,
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




validation_data_excel_list = []
#################start test after train#################
with open(dataset_test_path+'test.txt', 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    # pbar = tqdm(lines)
    real_count=0

    # j = 0

    for line in lines:
        # 控制处理几个任务
        # j += 1
        # if j>5:
        #     break

        print ('real_count',real_count)
        real=data_real[real_count]
        # real_json = json.loads(real.strip())
        # real_box_all = real_json['real']
        real_json = json.loads(real.strip())
        image_name=real_json['image']
        real_box_all = real_json['real']

        print ('\n================afterelearning=========================================================================================================================================================================================================================================================================================================================================================================================================================beforelearning=============================')
        image_path,instruction=line.split(';')
        instruction=instruction[:-1]
        print ('image_path:',image_path)
        print ('Instruction:',instruction)
        image = Image.open(dataset_test_path+image_path)

        image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        # 定义保存图片的完整路径，方便后续同时用于保存图片和写入Excel
        save_initial_img_complete_path = validation_result_save_path + instruction + '-validation_after_train.png'
        # 保存一下原图片的地址，方便后面查看
        print("将原图片保存到：=================================================",
              validation_result_save_path + instruction + '-validation_after_train.png')
        init_state['IMAGE'].save(save_initial_img_complete_path)


        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, after_extra_out, after_real_loc, test_prog_ppl, test_subq_ppl, VQA_word_probability =CLOVA_model.inference(instruction, init_state)
        save_final_img_complete_path = ""
        if can_run==False or after_extra_out==None:
            print("===========can_run==False or after_extra_out============")
            failed_prog=failed_prog+1
            result=''
            after_extra_out=[]
            cur_js= {}
            cur_js['image'] = image_path
            cur_js['res'] = after_extra_out
            after_ans_file.write(json.dumps(cur_js) + '\n')
            after_ans_file.flush()
            real_count=real_count+1

        else:
            cur_js= {}
            cur_js['image'] = image_path
            if len(after_extra_out) > 0:
                for data in after_extra_out:
                    if 'category' in data.keys():
                        del data['category']
                    if 'inst_id' in data.keys():
                        del data['inst_id']
                    if 'mask' in data.keys():
                        del data['mask']
                    if 'class_score' in data.keys():
                        del data['class_score']
            cur_js['res'] = after_extra_out
            after_ans_file.write(json.dumps(cur_js) + '\n')
            after_ans_file.flush()

            # 最后程序执行之后得到的图片

            if 'Image' in str(type(result)):
                save_final_img_complete_path = result_save_path + image_path + instruction + 'validation_after_train_result.png'
                result.save(save_final_img_complete_path)
                # 无论程序运行是否成功，都尝试保存记录（如果运行失败，prog可能为空，视情况而定）

        validation_data_excel_list.append({
            "问题": instruction,
            "原图片": save_initial_img_complete_path,  # 这里存的是路径
            "标签": real_box_all,
            "验证阶段推理出来的子步骤": subq,
            "验证阶段推理出来的程序的": prog,
            "验证阶段推理结果": result,
            "推理结果的图片": save_final_img_complete_path,  # 这里存的是最后图片的
        })
# ================= 循环结束后的保存代码 =================
print("正在生成带真实图片的 Excel 表格...")

# 1. 创建 DataFrame
df = pd.DataFrame(validation_data_excel_list)

# 2. 定义文件名
output_file = "validation_KNOWTAG_list.xlsx"
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

    # ================= 插入图片的循环 =================
    print("正在逐行插入图片...")

    # 遍历每一行数据
    for idx, row in df.iterrows():
        initial_img_path = row['原图片']  # 获取路径
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











