from safetensors import safe_open
import json

# *** 把这里改成你的完整文件路径 ***
file_path = "D:/Users/Mr.F/Downloads/model.safetensors"

print(f"Attempting to read metadata from: {file_path}")

try:
    with safe_open(file_path, framework="pt") as f:
        # metadata() 方法就是用来读取JSON头部的
        metadata = f.metadata()

        if metadata:
            print("\n✅ Success! File header (metadata) is NOT empty:")
            # 打印JSON头部内容
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
        else:
            print("\n⚠️ Warning: File header is empty (no metadata found).")

except Exception as e:
    print(f"\n❌ FAILED to read file:")
    print(f"Error: {e}")
    print("\nThis confirms the file is corrupted, empty, or not a valid safetensors file.")