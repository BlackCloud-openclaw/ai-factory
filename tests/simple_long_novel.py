#!/usr/bin/env python
"""
AI Factory 长篇小说生成脚本
支持断点续写，防止重复执行
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

# ========== 日志函数 ==========
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

# ========== 锁机制：防止重复执行 ==========
def acquire_lock():
    """获取文件锁，防止脚本重复运行"""
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
    """释放文件锁"""
    global lock_fd
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        log("释放执行锁")
    except:
        pass

# ========== API 调用封装 ==========
def wait_for_task(task_id, timeout=3600, poll_interval=60):
    """等待异步任务完成"""
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

def sync_request(endpoint, payload, timeout=1800):
    """同步请求（阻塞等待）"""
    try:
        resp = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json()
        else:
            log(f"请求失败: {resp.status_code} - {resp.text}", "ERROR")
            return False, None
    except Exception as e:
        log(f"请求异常: {e}", "ERROR")
        return False, None

def async_request(endpoint, payload):
    """异步请求（返回 task_id）"""
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

# ========== 检查小说状态 ==========
def check_novel_status():
    """检查小说是否有已有进度"""
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/progress", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            current_chapter = data.get("current_chapter", 1)
            current_volume = data.get("current_volume", 1)
            log(f"小说当前进度: 第{current_volume}卷第{current_chapter}章")
            return current_chapter > 1 or current_volume > 1
        elif resp.status_code == 404:
            log("小说不存在，将从头开始")
            return False
        else:
            log(f"查询进度失败: {resp.status_code}", "WARNING")
            return False
    except Exception as e:
        log(f"查询进度异常: {e}", "WARNING")
        return False

# ========== 主流程 ==========
def main():
    log("=" * 50)
    log("AI Factory 长篇小说生成脚本启动")
    log(f"Novel ID: {NOVEL_ID}")
    log("=" * 50)
    
    # 1. 获取执行锁
    if not acquire_lock():
        sys.exit(1)
    
    try:
        # 2. 检查小说状态
        has_progress = check_novel_status()
        
        # 3. 生成大纲（如果没有）
        if not has_progress:
            log("步骤1: 生成小说大纲...")
            outline_payload = {
                "user_input": "写一部修仙小说，共5卷，每卷10章，每章3个场景。主角林逸，从炼气期飞升。",
                "task_type": "novel_outline",
                "novel_id": NOVEL_ID,
                "resume": False
            }
            success, result = sync_request("execute", outline_payload, timeout=1800)
            if not success:
                log("大纲生成失败，退出", "ERROR")
                sys.exit(1)
            log("大纲生成成功")
            time.sleep(2)  # 等待数据库写入完成
        else:
            log("跳过大纲生成（小说已有进度）")
        
        # 4. 开始/继续写作
        log("步骤2: 开始/继续写作...")
        writing_payload = {
            "user_input": "开始写作，自动完成所有5卷10章，每章3个场景。",
            "task_type": "scene_plan",
            "novel_id": NOVEL_ID,
            "resume": has_progress  # 如果有进度，使用续写模式
        }
        
        if has_progress:
            log("使用断点续写模式 (resume=true)")
        else:
            log("使用全新写作模式")
        
        # 使用异步接口，支持长时间运行
        success, task_id = async_request("resume", writing_payload)
        if not success or not task_id:
            log("启动写作任务失败", "ERROR")
            sys.exit(1)
        
        # 5. 等待任务完成
        log("等待写作任务完成（可能需要数小时）...")
        success, result = wait_for_task(task_id, timeout=86400)  # 24小时超时
        
        if success:
            log("=" * 50)
            log("✅ 小说生成完成！")
            log(f"小说 ID: {NOVEL_ID}")
            log(f"查看文件: data/novels/{NOVEL_ID}/")
            log("=" * 50)
        else:
            log("❌ 小说生成失败", "ERROR")
            sys.exit(1)
            
    except KeyboardInterrupt:
        log("用户中断", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"脚本异常: {e}", "ERROR")
        sys.exit(1)
    finally:
        release_lock()

# ========== 添加 /novel_id/{novel_id}/progress 端点（如果没有的话）==========
# 注意：这个端点可能需要你在 API 中添加
# 临时方案：使用 writing_progress 表查询
def add_progress_endpoint_if_needed():
    """检查并提示添加进度查询端点"""
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/progress", timeout=5)
        if resp.status_code == 404:
            log("提示: API 缺少 /novel_id/{novel_id}/progress 端点", "WARNING")
            log("将使用备用方案（直接尝试写作）")
            return False
    except:
        pass
    return True

if __name__ == "__main__":
    main()