import os
# 设置国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 新增 Llama-2-7b 模型仓库
models_to_download = [
    "meta-llama/Llama-2-7b-hf",  # Llama-2 7B 模型（需申请权限）
    # "runwayml/stable-diffusion-inpainting",
    # "google/owlvit-base-patch32",
    # "openai/clip-vit-large-patch14"
]

# 模型保存目录（与你的服务器路径一致）
base_download_dir = "/home/fanchuanhua/project/CLOVA/model"
os.makedirs(base_download_dir, exist_ok=True)

print(f"开始下载模型到 {base_download_dir} 目录...")
print(f"当前使用的镜像源：{os.environ['HF_ENDPOINT']}")

# 替换为你的 Hugging Face Token（获取方式见下方）
HF_TOKEN = ""

for repo_id in models_to_download:
    print(f"\nDownloading: {repo_id}")
    local_dir_name = repo_id.split('/')[-1]
    local_path = os.path.join(base_download_dir, local_dir_name)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,  # 断点续传
            token=HF_TOKEN,  # 传入Token以访问需权限的模型
            max_workers=8  # 多线程加速
        )
        print(f"✅ 成功下载 {repo_id} 到 {local_path}")
    except Exception as e:
        print(f"❌ 下载 {repo_id} 失败: {str(e)[:200]}")

print("\n所有模型下载任务已完成（失败的模型请查看上方日志）。")