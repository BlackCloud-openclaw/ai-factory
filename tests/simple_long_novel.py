#!/usr/bin/env python
"""
AI Factory 长篇小说生成脚本（5卷 × 100章 = 500章版）
使用分卷 API 生成大纲，支持断点续传。
"""

import requests
import time
import json
import sys
import os
import fcntl
import re
from datetime import datetime

# ========== 配置 ==========
API_BASE = "http://localhost:8000/api/v1"
NOVEL_ID = "simple_long_novel_001"
LOCK_FILE = "/tmp/ai_factory_simple_long_novel.lock"

TOTAL_VOLUMES = 5
CHAPTERS_PER_VOLUME = 100

# 卷的默认信息（可根据需要手动调整）
VOLUME_INFO = {
    1: {"title": "初入修仙界", "target_realm": "筑基", "core_conflict": "入门之争与机缘争夺"},
    2: {"title": "金丹问道", "target_realm": "金丹", "core_conflict": "宗门内乱与灵脉争夺"},
    3: {"title": "元婴秘境", "target_realm": "元婴", "core_conflict": "上古遗迹与魔道围剿"},
    4: {"title": "化神劫", "target_realm": "化神", "core_conflict": "天道反噬与正邪决战"},
    5: {"title": "飞升路", "target_realm": "大乘", "core_conflict": "九重天劫与宿命对决"}
}

# ========== 日志函数 ==========
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

# ========== 锁机制 ==========
lock_fd = None
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
    if lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except:
            pass
        log("释放执行锁")

# ========== 大纲 API 调用 ==========
def get_outline_detail():
    """获取大纲详细内容"""
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/outline/detail", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("outline")
    except:
        pass
    return None

def get_volume_chapters_count(volume_num):
    """获取指定卷已有的章节数"""
    outline = get_outline_detail()
    if not outline:
        return 0
    volumes = outline.get("volumes", [])
    for vol in volumes:
        if vol.get("volume_num") == volume_num:
            return len(vol.get("chapters", []))
    return 0

def generate_volume_chapters(volume_num, volume_title, target_realm, core_conflict):
    """调用 API 生成指定卷的章节列表（超时 30 分钟）"""
    payload = {
        "novel_id": NOVEL_ID,
        "volume_num": volume_num,
        "total_chapters": CHAPTERS_PER_VOLUME,
        "volume_title": volume_title,
        "target_realm": target_realm,
        "core_conflict": core_conflict
    }
    try:
        resp = requests.post(f"{API_BASE}/novel/volume/chapters", json=payload, timeout=1800)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "already_complete":
                log(f"卷 {volume_num} 已完整，跳过")
                return True, CHAPTERS_PER_VOLUME
            elif data.get("status") == "success":
                log(f"卷 {volume_num} 生成成功，获得 {data.get('chapters_count')} 章")
                return True, data.get("chapters_count", 0)
            else:
                log(f"卷 {volume_num} 生成失败: {data}", "ERROR")
                return False, 0
        else:
            log(f"卷 {volume_num} 请求失败: {resp.status_code} - {resp.text}", "ERROR")
            return False, 0
    except requests.exceptions.Timeout:
        log(f"卷 {volume_num} 请求超时（30分钟），请检查后端日志", "ERROR")
        return False, 0
    except Exception as e:
        log(f"卷 {volume_num} 生成异常: {e}", "ERROR")
        return False, 0

def generate_all_volumes():
    """逐卷生成全部大纲（支持断点续传）"""
    for vol_num in range(1, TOTAL_VOLUMES + 1):
        existing_chapters = get_volume_chapters_count(vol_num)
        if existing_chapters >= CHAPTERS_PER_VOLUME:
            log(f"卷 {vol_num} 已有 {existing_chapters} 章，已完整，跳过")
            continue

        info = VOLUME_INFO.get(vol_num, {
            "title": f"第{vol_num}卷",
            "target_realm": "筑基",
            "core_conflict": "未知冲突"
        })
        log(f"开始生成卷 {vol_num}: {info['title']}...")
        success, count = generate_volume_chapters(
            vol_num, info["title"], info["target_realm"], info["core_conflict"]
        )
        if not success:
            log(f"卷 {vol_num} 生成失败，停止后续生成", "ERROR")
            return False
        if count != CHAPTERS_PER_VOLUME:
            log(f"卷 {vol_num} 生成章数不足：{count}/{CHAPTERS_PER_VOLUME}", "WARNING")
        time.sleep(2)  # 避免请求过快

    final_outline = get_outline_detail()
    if final_outline:
        total_chapters = sum(len(v.get("chapters", [])) for v in final_outline.get("volumes", []))
        log(f"✅ 所有卷生成完毕，总章节数: {total_chapters}")
        return True
    else:
        log("❌ 最终验证失败，大纲不存在", "ERROR")
        return False

# ========== 写作流程 API ==========
def get_progress():
    """获取当前写作进度（卷、章）"""
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/progress", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            vol = data.get("current_volume")
            ch = data.get("current_chapter")
            # 处理 None 值（数据库字段可能为 NULL）
            if vol is None:
                vol = 1
            if ch is None:
                ch = 1
            return vol, ch
        elif resp.status_code == 404:
            return None, None
        else:
            log(f"查询进度失败: {resp.status_code}", "WARNING")
            return None, None
    except Exception as e:
        log(f"查询进度异常: {e}", "WARNING")
        return None, None

def wait_for_task(task_id, timeout=7200, poll_interval=30):
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
                    progress = data.get('progress', 0)
                    if progress > 0 and progress % 20 == 0:
                        log(f"任务 {task_id} 进度: {progress}%")
            else:
                log(f"查询任务状态失败: {resp.status_code}", "WARNING")
        except Exception as e:
            log(f"查询任务状态异常: {e}", "WARNING")
        time.sleep(poll_interval)
    log(f"任务 {task_id} 超时", "ERROR")
    return False, None

def async_request(endpoint, payload):
    """启动异步任务"""
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

def is_novel_completed(volume, chapter):
    if volume is None or chapter is None:
        return False
    if volume > TOTAL_VOLUMES:
        return True
    if volume == TOTAL_VOLUMES and chapter > CHAPTERS_PER_VOLUME:
        return True
    return False

def fetch_latest_entropy():
    """从日志文件中提取最近一次叙事熵（用于监控）"""
    log_file = "logs/ai_factory.log"
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "Narrative entropy for volume" in line:
                match = re.search(r"local=([\d.]+), arc=([\d.]+), civ=([\d.]+)", line)
                if match:
                    return {
                        "local": float(match.group(1)),
                        "arc": float(match.group(2)),
                        "civ": float(match.group(3))
                    }
    except Exception as e:
        log(f"读取熵值失败: {e}", "WARNING")
    return None

# ========== 主流程 ==========
def main():
    log("=" * 60)
    log(f"AI Factory 长篇小说生成脚本（{TOTAL_VOLUMES}卷 × {CHAPTERS_PER_VOLUME}章 = {TOTAL_VOLUMES * CHAPTERS_PER_VOLUME}章）")
    log(f"Novel ID: {NOVEL_ID}")
    log("=" * 60)

    if not acquire_lock():
        sys.exit(1)

    try:
        # ========== 第一步：生成完整大纲（逐卷，支持断点） ==========
        log("检查大纲完整性...")
        if not generate_all_volumes():
            log("大纲生成失败，无法继续", "ERROR")
            sys.exit(1)
        log("✅ 大纲已完整，开始写作...")

        # ========== 第二步：逐章写作（原有逻辑） ==========
        start_vol, start_ch = get_progress()
        if start_vol is None:
            log("无法获取初始进度", "ERROR")
            sys.exit(1)
        log(f"从第{start_vol}卷第{start_ch}章开始续写")

        chapter_count = (start_vol - 1) * CHAPTERS_PER_VOLUME + (start_ch - 1)

        while True:
            vol, ch = get_progress()
            if is_novel_completed(vol, ch):
                log("🎉 所有500章已生成完毕！🎉")
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

            success, _ = wait_for_task(task_id, timeout=7200)
            if not success:
                log(f"第{vol}卷第{ch}章生成失败，终止", "ERROR")
                break

            chapter_count += 1
            log(f"✅ 第{vol}卷第{ch}章完成 (累计完成 {chapter_count} 章)")

            if chapter_count % 10 == 0:
                entropy = fetch_latest_entropy()
                if entropy:
                    log(f"📊 最新叙事熵 - local={entropy['local']:.3f}, arc={entropy['arc']:.3f}, civ={entropy['civ']:.3f}")
                else:
                    log("📊 未能获取熵值（日志文件可能尚未更新）")

            time.sleep(2)

        log("=" * 60)
        log("✅ 500章生成完成！")
        log(f"小说 ID: {NOVEL_ID}")
        log(f"查看文件: data/novels/{NOVEL_ID}/")
        log("=" * 60)

    except KeyboardInterrupt:
        log("用户中断", "WARNING")
    except Exception as e:
        log(f"脚本异常: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        release_lock()

if __name__ == "__main__":
    main()