#!/usr/bin/env python3
"""
Judge whether an image-editing task was completed correctly with Qwen-VL-Max.

Usage:
    python qwen_vl_edit_judge.py --original original.jpg --edited edited.jpg

Required environment variable:
    DASHSCOPE_API_KEY=your_api_key
"""

from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import httpx
import ruamel.yaml as yaml
import time

from openai import OpenAI

LLM_config_path='/home/fanchuanhua/project/CLOVA/CLOVA-tool/configs/LLM_config.yaml'
LLM_config= yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

default_model = LLM_config["Qwen"]["default_model"]
default_base_url = LLM_config["Qwen"]["base_url"]
default_api_key = LLM_config["Qwen"]["api_key"]
default_timeout = LLM_config["Qwen"]["time_out"]
server_proxy_url = LLM_config["Qwen"]["proxy_url"]


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_evaluation_prompt(request: str) -> str:
    return f"""You are an expert AI image editing evaluator.

Your task is to judge whether the Edited Image correctly satisfies the user's editing request by comparing the Original Image and the Edited Image.

User Editing Request:
"{request}"

Judging Rules:
1. Check whether the requested edit has been correctly completed.
2. The edited content must match the user's request in object, attribute, position, count, color, style, and spatial relationship when these are mentioned.
3. Unrelated regions should remain unchanged, including background, lighting, camera angle, image style, and other objects.
4. The edit should look natural and visually consistent with the original image.
5. Return "false" if the requested edit is missing, incomplete, wrong, unnatural, or if unrelated parts of the image are changed.
6. Return "true" only if the edited image fully satisfies the request and preserves unrelated content.

Output Requirement:
Output ONLY one word:
true
or
false

Do not output explanations, reasoning, punctuation, markdown, or any other text."""

def completions_endpoint(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


# def call_qwen_vl_max(
#     *,
#     api_key: str,
#     base_url: str,
#     model: str,
#     original_image: str,
#     edited_image: str,
#     request: str,
#     timeout: int,
# ) -> str:
#     original_data_url = image_to_data_url(original_image)
#     edited_data_url = image_to_data_url(edited_image)
#     prompt = build_evaluation_prompt(request)
#
#     payload = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Original Image:"},
#                     {"type": "image_url", "image_url": {"url": original_data_url}},
#                     {"type": "text", "text": "Edited Image:"},
#                     {"type": "image_url", "image_url": {"url": edited_data_url}},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ],
#         "temperature": 0,
#         "max_tokens": 16,
#     }
#
#     body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
#     request_obj = urllib.request.Request(
#         completions_endpoint(base_url),
#         data=body,
#         headers={
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json",
#         },
#         method="POST",
#     )
#
#     try:
#         with urllib.request.urlopen(request_obj, timeout=timeout) as response:
#             response_body = response.read().decode("utf-8")
#     except urllib.error.HTTPError as exc:
#         error_body = exc.read().decode("utf-8", errors="replace")
#         raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
#     except urllib.error.URLError as exc:
#         raise RuntimeError(f"Request failed: {exc}") from exc
#
#     data = json.loads(response_body)
#     try:
#         return data["choices"][0]["message"]["content"]
#     except (KeyError, IndexError, TypeError) as exc:
#         raise RuntimeError(f"Unexpected API response: {response_body}") from exc
def call_qwen_vl_max(
        *,
        api_key: str,
        base_url: str,
        model: str,
        original_image: str,
        edited_image: str,
        request: str,
        timeout: int,
) -> str:
    print(f"[Debug] 开始处理图片转 Base64...")

    original_data_url = image_to_data_url(original_image)
    edited_data_url = image_to_data_url(edited_image)

    print(f"[Debug] 原图 Base64 长度: {len(original_data_url)}")
    print(f"[Debug] 编辑图 Base64 长度: {len(edited_data_url)}")

    prompt = build_evaluation_prompt(request)

    # 获取服务器系统中配置的代理（终端 curl 能通说明这里有值）
    # 如果没取到，默认退回本地常见代理端口作为双保险
    # proxy_url = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:7890"
    proxy_url = server_proxy_url
    print(f"[Debug] 当前使用网络通信代理: {proxy_url}")
    # 显式构造底层 http 客户端，绑定系统代理，并设置严谨的握手和读取超时
    # 这能彻底避免 requests/urllib 在发送几百 KB 图像包时的死锁问题
    custom_http_client = httpx.Client(
        proxies=proxy_url,
        timeout=httpx.Timeout(float(timeout), connect=15.0),  # 连接超时 15 秒，总超时取你的配置文件
        # 增加超时时间：视觉大模型处理大图片需要更长时间，连接20秒，读取120秒
        # timeout = httpx.Timeout(120.0, connect=20.0),
        # # 【关键】关闭长连接池 (Keep-Alive)，强制每次请求建立全新连接，防止代理意外断开导致报错
        # limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0),
        # # 增加底层自动重试次数
        # transport = httpx.HTTPTransport(retries=3, local_address="0.0.0.0")
    )
    # print("[Debug] client.chat 向 base_url 发送请求:",base_url)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,  # 注意：这里应该是 https://dashscope.aliyuncs.com/compatible-mode/v1
        http_client=custom_http_client # 传入定制客户端，建立代理加密隧道
    )

    try:
        print(f"[Debug] client.chat 向 base_url 发送请求: {base_url}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original Image:"},
                        {"type": "image_url", "image_url": {"url": original_data_url}},
                        {"type": "text", "text": "Edited Image:"},
                        {"type": "image_url", "image_url": {"url": edited_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0,
            max_tokens=16,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"调用 Qwen 模型失败: {e}") from e

def normalize_boolean_output(model_output: str) -> str:
    text = model_output.strip().lower()
    text = text.strip("`'\" \t\r\n.")

    if text in {"true", "false"}:
        return text

    matches = re.findall(r"\b(true|false)\b", text)
    if len(matches) == 1:
        return matches[0]

    raise ValueError(f'Model did not return a clear "true" or "false": {model_output!r}')


# 调用直接回输出结果
def judge_the_result_of_imagedit(original_image, edited_image, request):
    """
    original_image: 输入图片
    edited_image: 我的系统输出的图片
    request: 请求
    """
    try:
        raw_output = call_qwen_vl_max(
            original_image=original_image,
            edited_image=edited_image,
            request=request,
            api_key=default_api_key,
            base_url=default_base_url,
            model=default_model,
            timeout=default_timeout,
        )
        # timeout: 请求的超时时间。因为视觉大模型处理图片可能较慢，所以默认设置了较长的 2 分钟。
        result = normalize_boolean_output(raw_output)
        print("\n===================本次图像编辑任务的结果正确与否：===============\n", result)
        print("===================本次图像编辑任务的结果正确与否：===============\n")
        return result
    except Exception as exc:
        print(f"调用 Qwen 模型出错了，Error: {exc}", file=sys.stderr)
