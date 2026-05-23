#!/usr/bin/env python
"""
AI Factory 长篇小说生成脚本（完整版）
支持自动续写全部5卷50章，每章3个场景。
需要 API 提供 /novel_id/{novel_id}/progress 端点。
"""

import requests
import time
import json
import sys
import os
import fcntl
from datetime import datetime

# ========== 配置 ==========
API_BASE = "http://localhost:8000/api/v1"
NOVEL_ID = "simple_long_novel_001"
LOCK_FILE = "/tmp/ai_factory_simple_long_novel.lock"

TOTAL_VOLUMES = 5
CHAPTERS_PER_VOLUME = 10

# ========== 日志函数 ==========
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

# ========== 锁机制 ==========
def acquire_lock():
    global lock_fd
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log("成功获取执行锁")
        return True
    except IOError:
        log("另一个脚本实例正在运行，退出", "WARNING")
        return False

def release_lock():
    global lock_fd
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        log("释放执行锁")
    except:
        pass

# ========== API 调用 ==========
def wait_for_task(task_id, timeout=3600, poll_interval=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{API_BASE}/task/{task_id}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status == "success":
                    log(f"任务 {task_id} 完成")
                    return True, data
                elif status == "failed":
                    log(f"任务 {task_id} 失败: {data.get('error')}", "ERROR")
                    return False, data
                else:
                    log(f"任务 {task_id} 状态: {status}, 进度: {data.get('progress', 0)}%")
            else:
                log(f"查询任务状态失败: {resp.status_code}", "WARNING")
        except Exception as e:
            log(f"查询任务状态异常: {e}", "WARNING")
        time.sleep(poll_interval)
    log(f"任务 {task_id} 超时", "ERROR")
    return False, None

def async_request(endpoint, payload):
    try:
        resp = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            task_id = data.get("task_id")
            if task_id:
                log(f"异步任务已创建: {task_id}")
                return True, task_id
            else:
                log(f"响应中无 task_id: {data}", "WARNING")
                return False, None
        else:
            log(f"请求失败: {resp.status_code} - {resp.text}", "ERROR")
            return False, None
    except Exception as e:
        log(f"请求异常: {e}", "ERROR")
        return False, None

def get_progress():
    """获取当前写作进度（卷、章）"""
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/progress", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("current_volume", 1), data.get("current_chapter", 1)
        elif resp.status_code == 404:
            return None, None
        else:
            log(f"查询进度失败: {resp.status_code}", "WARNING")
            return None, None
    except Exception as e:
        log(f"查询进度异常: {e}", "WARNING")
        return None, None

def is_novel_completed(volume, chapter):
    if volume is None or chapter is None:
        return False
    if volume > TOTAL_VOLUMES:
        return True
    if volume == TOTAL_VOLUMES and chapter > CHAPTERS_PER_VOLUME:
        return True
    return False

def generate_outline():
    """生成小说大纲（同步）"""
    log("生成小说大纲...")
    payload = {
        "user_input": "写一部修仙小说，共5卷，每卷10章，每章3个场景。主角林逸，从炼气期飞升。",
        "task_type": "novel_outline",
        "novel_id": NOVEL_ID,
        "resume": False
    }
    try:
        resp = requests.post(f"{API_BASE}/execute", json=payload, timeout=7200)
        if resp.status_code == 200:
            log("大纲生成成功")
            return True
        else:
            log(f"大纲生成失败: {resp.status_code}", "ERROR")
            return False
    except Exception as e:
        log(f"大纲生成异常: {e}", "ERROR")
        return False

# ========== 主流程 ==========
def main():
    log("=" * 50)
    log("AI Factory 长篇小说生成脚本启动（完整版）")
    log(f"Novel ID: {NOVEL_ID}")
    log(f"目标: {TOTAL_VOLUMES}卷 × {CHAPTERS_PER_VOLUME}章 = {TOTAL_VOLUMES * CHAPTERS_PER_VOLUME}章")
    log("=" * 50)

    if not acquire_lock():
        sys.exit(1)

    try:
        # 检查是否有大纲（通过进度是否存在判断）
        vol, ch = get_progress()
        if vol is None:
            log("小说尚未初始化，开始生成大纲...")
            if not generate_outline():
                sys.exit(1)
            time.sleep(2)  # 等待数据库写入
            vol, ch = get_progress()
            if vol is None:
                log("大纲生成后仍无法获取进度，请检查 API 端点 /progress", "ERROR")
                sys.exit(1)

        # 循环写作直到完成
        while True:
            vol, ch = get_progress()
            if is_novel_completed(vol, ch):
                log("🎉 所有章节已生成完毕！🎉")
                break

            log(f"开始生成第{vol}卷第{ch}章...")
            payload = {
                "user_input": "继续写作",
                "task_type": "scene_plan",
                "novel_id": NOVEL_ID,
                "resume": True
            }
            success, task_id = async_request("resume", payload)
            if not success or not task_id:
                log("启动写作任务失败", "ERROR")
                break

            success, _ = wait_for_task(task_id, timeout=3600)
            if not success:
                log(f"第{vol}卷第{ch}章生成失败，终止", "ERROR")
                break

            log(f"✅ 第{vol}卷第{ch}章完成")
            time.sleep(2)

        log("=" * 50)
        log("✅ 小说生成完成！")
        log(f"小说 ID: {NOVEL_ID}")
        log(f"查看文件: data/novels/{NOVEL_ID}/")
        log("=" * 50)

    except KeyboardInterrupt:
        log("用户中断", "WARNING")
    except Exception as e:
        log(f"脚本异常: {e}", "ERROR")
        sys.exit(1)
    finally:
        release_lock()

if __name__ == "__main__":
    main()