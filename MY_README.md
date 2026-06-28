# /home/fanchuanhua/project/CLOVA/CLOVA-tool  整个项目在服务器里面的地址
# CLOVA Tool Project README
本项目基于 CLOVA: A Closed-Loop Visual Assistant with Tool Usage and Update，是一个面向视觉任务的闭环工具调用与工具更新框架。项目核心思想是让大语言模型先把视觉问题拆解为子问题和可执行程序，再调用视觉工具完成推理；如果结果不正确，则进入反思和学习阶段，尝试修正程序或更新工具。

当前工作区在原始 CLOVA 代码基础上加入了本地实验脚本、图像编辑评估、知识标注评估、Qwen-VL 判别、Excel 结果导出等内容。

## 1. 项目核心流程

CLOVA 的主流程分为三步：

1. Inference：根据输入问题生成子步骤和程序，并调用工具执行。
2. Reflection：根据人工反馈或程序执行状态判断错误来源。
3. Learning：如果需要，更新经验池或视觉工具，使后续任务表现更好。

主要入口在：

```text
framework/clova.py
```

其中 `CLOVA.inference()` 负责推理，`CLOVA.reflection()` 负责反思，`CLOVA.learning()` 负责学习与工具更新。

## 2. 目录结构

```text
.
├── configs/
│   ├── LLM_config.yaml                 # LLM、任务类型、数据路径、结果路径配置
│   └── all_updated_model_config.yaml   # 各视觉工具和模型权重路径配置
├── framework/
│   └── clova.py                        # CLOVA 主框架
├── engine/
│   ├── utils.py                        # ProgramGenerator、ProgramInterpreter 等核心执行逻辑
│   ├── step_interpreters.py            # 根据任务类型注册可用工具
│   └── data_utils.py                   # 数据处理辅助函数
├── tools/
│   ├── LOC.py                          # 开放词汇目标定位工具
│   ├── SEG.py                          # 分割工具
│   ├── SELECT.py                       # 图像区域选择工具
│   ├── REPLACE.py                      # 图像替换/编辑工具
│   ├── CLASSIFY.py                     # 知识标注分类工具
│   ├── VQA.py                          # VQA 工具
│   ├── FaceDet.py                      # 人脸检测工具
│   ├── unupdated_functions.py          # COUNT、CROP、TAG、RESULT 等非学习型工具
│   └── model_updating.py               # 工具更新逻辑
├── prompts/
│   ├── prompt_engineering.py           # 工具说明、提示词生成逻辑
│   ├── *_experience_pool.py            # 不同任务的 in-context 示例与经验池
│   └── intermediate_result.py          # 中间结果转文本逻辑
├── image_editing_data/
│   ├── train_data.txt                  # 图像编辑训练数据
│   ├── test_data.txt                   # 图像编辑测试数据
│   └── images/                         # 图像编辑图片
├── knowtag_data/
│   ├── knowledge_tagging_dataset_v5_train/
│   └── knowledge_tagging_dataset_v5_test/
├── my_tool/
│   └── Qwen_invoke/
│       └── qwen_vl_edit_judge.py       # Qwen-VL 图像编辑结果判别
├── imgedit_demo.py                     # 图像编辑实验入口
├── knowtag_demo.py                     # 知识标注实验入口
├── gqa_demo.py     # GQA 相关实验脚本
├── whole_test_prompt_regenetate.py                # 根据我们自己设计的 prompt 生成出错原因，然后重新生成程序，可以判断出错原因好坏的代码。
├── environment.yml                     # Conda 环境配置
└── README.md                           # 原始项目 README
```

## 3. 环境安装

建议使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate clova_tool_capacity
```

当前 `environment.yml` 使用 Python 3.10.12、PyTorch 2.0.1、CUDA 11.8，并安装了 `transformers`、`diffusers`、`openai`、`pandas`、`xlsxwriter` 等依赖。

如果 Excel 导出时报错找不到 `xlsxwriter`，可以补装：

```bash
pip install xlsxwriter
```

## 4. 重要配置文件

### 4.1 `configs/LLM_config.yaml`

这个文件控制任务类型、LLM 配置、Qwen 配置、数据路径和结果保存路径。

关键字段：

```yaml
LLaMA:
  ckpt_dir_path: ...
  tokenizer_path: ...

OpenAI:
  base_url: ...
  api_key: ...

Qwen:
  default_model: ...
  base_url: ...
  api_key: ...
  time_out: ...
  proxy_url: ...

Task_type: 'imgedit'
```

`Task_type` 决定 `ProgramInterpreter` 注册哪些工具：

```yaml
Task_type: 'gqa'
Task_type: 'nlvr'
Task_type: 'imgedit'
Task_type: 'knowtag'
```

运行 `imgedit_demo.py` 前应设置为：

```yaml
Task_type: 'imgedit'
```

运行 `knowtag_demo.py` 前应设置为：

```yaml
Task_type: 'knowtag'
```

### 4.2 `configs/all_updated_model_config.yaml`

这个文件控制各视觉工具的模型路径与训练参数。例如：

```yaml
LOC:
  init:
    pretrained_processor: ...
    pretrained_model: ...

SEG:
  init:
    pretrained_fe: ...
    pretrained_model: ...

REPLACE:
  init:
    sd_pipe:
      pretrained: ...
    pipe:
      pretrained: ...
```

如果运行时报模型文件找不到，优先检查这个文件里的本地路径是否存在。

## 5. 主要运行脚本

### 5.1 图像编辑任务

入口：
在命令行运行时
需要先把 import os 下面的几行给注释掉
```bash
CUDA_VISIBLE_DEVICES="0" nohup torchrun --nproc_per_node=1  --rdzv_endpoint=localhost:29501 imgedit_demo.py > clova_imgedit_demo_run.log 2>&1 &

tail -f clova_imgedit_demo_run.log
```
在pycharm里面右键运行时
需要先打开 import os 下面的几行，不注释那几行
然后直接右键运行就可以

功能：

1. 读取 `image_editing_data/test_data.txt`。
2. 对测试图片执行图像编辑任务。
3. 保存原图、编辑结果图。
4. 调用 Qwen-VL 判断编辑结果是否符合指令。
5. 将问题、原图、目标图、程序、结果图、Qwen 判别结果写入 Excel。
6. 进入训练阶段，使用反馈进行反思和学习。
7. 再次测试并生成验证阶段 Excel。

常见输出文件：

```text
test_IMAGEDIT_list.xlsx
validation_IMGEDIT_list.xlsx
image_editing_data/results/
IMGEDIT_result/new_test_result/
IMGEDIT_result/train_result/
IMGEDIT_result/validation_result/
```

### 5.2 知识标注任务

入口：

在命令行运行时
需要先把 import os 下面的几行给注释掉
```bash
CUDA_VISIBLE_DEVICES="0" nohup torchrun --nproc_per_node=1  --rdzv_endpoint=localhost:29501 knowtag_demo.py > clova_knowtag_demo_run.log 2>&1 &

tail -f clova_knowtag_demo_run.log
```
在pycharm里面右键运行时
需要先打开 import os 下面的几行，不注释那几行
然后直接右键运行就可以

功能：

1. 读取 `knowtag_data/knowledge_tagging_dataset_v5_test/test.txt`。
2. 对图片进行目标定位、类别列表生成、分类、标注可视化。
3. 保存原图和结果图。
4. 导出测试前和验证后的 Excel。
5. 训练阶段根据预测和标注计算 F1、precision、recall，并进行反思学习。

常见输出文件：

```text
test_KNOWTAG_list.xlsx
validation_KNOWTAG_list.xlsx
knowtag_data/results/
KNOWTAG_result/new_test_result/
KNOWTAG_result/train_result/
KNOWTAG_result/validation_result/
```

### 5.3 GQA 相关脚本

当前工作区包含：

```text
gqa_demo.py
```

入口：

在命令行运行时
需要先把 import os 下面的几行给注释掉(可以保存下来日志)
```bash
CUDA_VISIBLE_DEVICES="0" nohup torchrun --nproc_per_node=1  --rdzv_endpoint=localhost:29501 gqa_demo.py > clova_gqa_demo_run.log 2>&1 &

tail -f clova_gqa_demo_run.log
```
在pycharm里面右键运行时
需要先打开 import os 下面的几行，不注释那几行
然后直接右键运行就可以

这些脚本用于 GQA 视觉问答、工具能力测试或历史备份实验。运行前需要将 `configs/LLM_config.yaml` 中的 `Task_type` 改为 `gqa`，并确认 GQA 数据集路径正确。

## 6. Qwen-VL 图像编辑评估
这部分的代码是用 Qwen 大模型来判断图像编辑任务最后编辑之后的图片，是否正确完成了请求
已经将函数调用添加到imgedit_demo.py文件里面了
Qwen 调用逻辑在：

```text
my_tool/Qwen_invoke/qwen_vl_edit_judge.py
```

`imgedit_demo.py` 中通过以下函数调用：

```python
judge_the_result_of_imagedit(
    save_initial_img_complete_path,
    save_final_img_complete_path,
    question
)
```

该函数会：

1. 将原图和编辑图转为 base64。
2. 使用 OpenAI 兼容接口请求 DashScope Qwen。
3. 要求模型返回 `true` 或 `false`。

相关配置在 `configs/LLM_config.yaml`：

```yaml
Qwen:
  default_model: '<vision-model-name>'
  base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1'
  api_key: '<your-api-key>'
  time_out: 300
```

图像编辑结果判别需要使用支持图片输入的视觉模型，例如 `qwen-vl-plus` 或 `qwen-vl-max`。如果这里配置成纯文本模型，接口即使能连通，也不能正确完成两张图片的比较。

如果请求一直超时，优先检查：

1. 服务器是否能访问 `dashscope.aliyuncs.com`。
2. `base_url` 是否和 API Key 所属地域一致。
3. 是否被 `HTTP_PROXY` 或 `HTTPS_PROXY` 代理影响。
4. `qwen_vl_edit_judge.py` 是否读取的是你实际修改的配置文件。
5. 图片路径是否真实存在，最终图是否成功保存。

### 解决代理问题
如果服务器设置了代理，但 DashScope 需要直连，可以在导入 Qwen 调用模块前设置：
https://evtlsh4tsfb.feishu.cn/wiki/LsnYwoXP1iGiECkDfcYcBoD6nfh?from=from_copylink


## 7. Excel 结果导出

图像编辑和知识标注脚本都会将实验记录写入 Excel。

典型列包括：

```text
问题
原图片
标签或目标图
推理出来的子步骤
推理出来的程序
推理结果
推理结果的图片
大模型判断结果
```

导出逻辑一般为：

1. 将每条样本追加到 Python list。
2. 用 `pandas.DataFrame` 转成表格。
3. 使用 `pd.ExcelWriter(..., engine='xlsxwriter')` 写入 Excel。
4. 使用 `worksheet.insert_image(...)` 插入图片。

如果 Excel 为空，通常说明追加到 list 的数据为空。  
如果文字列有内容但图片列为空，通常说明图片路径为空、图片文件不存在，或者 `insert_image` 捕获了异常。

## 8. 工具系统

工具注册入口：

```text
engine/step_interpreters.py
```

不同任务注册的工具不同：

```text
gqa:
  LOC, COUNT, CROP, VQA, EVAL, RESULT

imgedit:
  FACEDET, SEG, SELECT, COLORPOP, BGBLUR, REPLACE, EMOJI, RESULT

knowtag:
  FACEDET, LIST, CLASSIFY, RESULT, TAG, LOC
```

工具实现主要位于 `tools/`。

可学习工具一般需要实现：

```python
execute(...)
update(...)
```

非学习工具一般放在：

```text
tools/unupdated_functions.py
```

例如 `COUNT`、`CROP`、`TAG`、`RESULT`。

## 9. 数据格式

### 9.1 图像编辑数据

测试数据一般形如：

```text
source_image;target_image;instruction
```

训练数据一般形如：

```text
source_image;target_image;instruction;correct;feedback
```

图片目录由 `configs/LLM_config.yaml` 中的字段控制：

```yaml
IMGEDIT:
  image_path: './image_editing_data/images/'
```

### 9.2 知识标注数据

测试数据一般形如：

```text
image_name;instruction
```

标注文件是 JSON Lines 格式，包含图片名和真实框：

```json
{"image":"xxx.jpg", "real":[{"box":[x1,y1,x2,y2], "class":"class name"}]}
```

## 10. 推荐运行顺序

第一次调试时建议只跑少量样本：

1. 确认 `Task_type`。
2. 确认数据路径和模型路径。
3. 确认输出目录存在。
4. 先只跑测试阶段 1 条样本。
5. 确认原图、结果图、Excel 都能生成。
6. 再打开训练和验证阶段。

## 得到 VQA 最后输出的 softmax 层概率的程序文件
tools/blip_vqa/blip_vqa.py 文件里面的 forward 函数

## 得到子步骤和程序的 PPL 困惑度的程序文件
engine/utils.py 里面 ProgramGenerator类的 generate 函数

## 得到输入 VQA 工具的问题和图片之间关系的热力图的程序文件
在 my_tool 那个文件夹里面的 tools/blip_vqa/blip_vqa.py 文件的 BLIP_VQA 类里面的 visualize_attention 函数来实现的
