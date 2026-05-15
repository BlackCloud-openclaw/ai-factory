#!/usr/bin/env python
import requests
import time
import json

API_BASE = "http://localhost:8000/api/v1"
NOVEL_ID = "simple_long_novel_001"

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# 1. 生成大纲
log("生成大纲中...")
outline_payload = {
    "user_input": "写一部修仙小说，共5卷，每卷10章，每章3个场景。主角林逸，从炼气期飞升。",
    "task_type": "novel_outline",
    "novel_id": NOVEL_ID,
    "resume": False
}
resp = requests.post(f"{API_BASE}/execute", json=outline_payload, timeout=1800)
if resp.status_code != 200:
    log(f"大纲生成失败: {resp.text}")
    exit(1)
log("大纲生成成功")

# 2. 开始写作
log("开始写作...")
writing_payload = {
    "user_input": "开始写作，自动完成所有5卷10章，每章3个场景。",
    "task_type": "scene_plan",
    "novel_id": NOVEL_ID,
    "resume": False
}
resp = requests.post(f"{API_BASE}/execute", json=writing_payload, timeout=86400)  # 24小时超时
if resp.status_code != 200:
    log(f"写作失败: {resp.text}")
    exit(1)
log("写作完成")