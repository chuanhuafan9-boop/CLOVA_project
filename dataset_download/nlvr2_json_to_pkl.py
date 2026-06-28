import json
import pickle
import os
import ruamel.yaml as yaml
from tqdm import tqdm


def get_image_filenames_from_identifier(identifier):
    try:
        parts = identifier.split('-')
        base_filename = '-'.join(parts[:-1])
        left_image_filename = f"{base_filename}-img0.png"
        right_image_filename = f"{base_filename}-img1.png"
        return left_image_filename, right_image_filename
    except Exception as e:
        print(f"处理 identifier 时出错: {identifier} - {e}")
        return None, None


def convert_json_to_pkl(json_path, pkl_path, file_desc):
    processed_dataset = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"读取 {file_desc}"):
                original_data = json.loads(line)

                left_img_name, right_img_name = get_image_filenames_from_identifier(original_data['identifier'])

                if left_img_name is None:
                    continue

                new_data = {
                    'sentence': original_data['sentence'],
                    'label': original_data['label'],
                    'identifier': original_data['identifier'],
                    'left_image': left_img_name,
                    'right_image': right_img_name,
                    'directory': original_data.get('directory')
                }
                processed_dataset.append(new_data)

        print(f"找到 {len(processed_dataset)} 个样本。")
        print(f"正在保存到 {pkl_path}...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(processed_dataset, f)
        print(f"{file_desc} .pkl 文件创建成功。")

    except FileNotFoundError:
        print(f"错误：找不到 {json_path}")
        print("请确保您的 .json 文件位于 .../nlvr2/data/ 目录中")
    except Exception as e:
        print(f"处理 {json_path} 时发生未知错误: {e}")


# --- 脚本主程序 ---
if __name__ == "__main__":
    print("开始转换 NLVR2 数据集...")

    # !!! 关键更改：我们在这里硬编码路径 !!!

    # 1. 定义 .json 文件的读取位置 (旧路径)
    json_base_path = '/home/fanchuanhua/project/CLOVA/clova_project/dataset_download/nlvr2-111/data/'  #

    # 2. 定义 .pkl 文件的保存位置 (新路径)
    pkl_base_path = '/home/fanchuanhua/project/CLOVA/clova_project/dataset_download/nlvr2/'  # (来自步骤 1)

    # 3. 定义输入/输出路径
    json_train_path = os.path.join(json_base_path, 'train.json')
    json_dev_path = os.path.join(json_base_path, 'dev.json')

    pkl_train_path = os.path.join(pkl_base_path, 'nlvr2_train_500_lambda.pkl')
    pkl_dev_path = os.path.join(pkl_base_path, 'nlvr2_dev_500_lambda.pkl')

    # 4. --- 处理 train.json ---
    convert_json_to_pkl(json_train_path, pkl_train_path, "train.json")

    # 5. --- 处理 dev.json ---
    convert_json_to_pkl(json_dev_path, pkl_dev_path, "dev.json")

    print("\n转换完成。")