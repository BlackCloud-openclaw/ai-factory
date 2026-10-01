#!/usr/bin/env python
"""
AI Factory 长篇小说生成脚本（5卷 × 100章 = 500章版）
使用分卷 API 生成大纲，支持断点续传。
修改版：自动修复进度不一致，支持连续写作。
"""

import requests
import time
import json
import sys
import os
import fcntl
import re
import psycopg2
from datetime import datetime
from pathlib import Path

# ========== 配置 ==========
API_BASE = "http://localhost:8000/api/v1"
NOVEL_ID = "simple_long_novel_001"
LOCK_FILE = "/tmp/ai_factory_simple_long_novel.lock"

TOTAL_VOLUMES = 5
CHAPTERS_PER_VOLUME = 100

# 数据库配置（从 .env 读取或直接写）
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "ai_factory")
DB_USER = os.getenv("POSTGRES_USER", "woami")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "kali")

# 卷的默认信息
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

# ========== 数据库连接（用于自动修复） ==========
def get_db_conn():
    """获取数据库连接（用于自动修复进度）"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        log(f"数据库连接失败: {e}", "ERROR")
        return None

def force_advance_chapter(vol, ch):
    """
    强制将进度推进到下一章。
    当章节文件已存在但进度未完成时调用。
    """
    conn = get_db_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        new_ch = ch + 1
        new_vol = vol
        
        # 检查是否超过当前卷
        total_ch = CHAPTERS_PER_VOLUME
        if new_ch > total_ch:
            new_vol = vol + 1
            new_ch = 1
        
        # 更新进度
        cur.execute("""
            UPDATE writing_progress 
            SET current_volume = %s, current_chapter = %s, current_scene = 0, chapter_completed = false, last_updated = NOW()
            WHERE project_id = %s
        """, (new_vol, new_ch, NOVEL_ID))
        conn.commit()
        cur.close()
        conn.close()
        log(f"✅ 自动修复进度: 第{vol}卷第{ch}章 → 第{new_vol}卷第{new_ch}章")
        return True
    except Exception as e:
        log(f"❌ 强制推进失败: {e}", "ERROR")
        return False

# ========== 大纲 API 调用 ==========
def get_outline_detail():
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/outline/detail", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("outline")
    except:
        pass
    return None

def get_volume_chapters_count(volume_num):
    outline = get_outline_detail()
    if not outline:
        return 0
    volumes = outline.get("volumes", [])
    for vol in volumes:
        if vol.get("volume_num") == volume_num:
            return len(vol.get("chapters", []))
    return 0

def generate_volume_chapters(volume_num, volume_title, target_realm, core_conflict):
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
            log(f"卷 {volume_num} 请求失败: {resp.status_code}", "ERROR")
            return False, 0
    except requests.exceptions.Timeout:
        log(f"卷 {volume_num} 请求超时", "ERROR")
        return False, 0
    except Exception as e:
        log(f"卷 {volume_num} 生成异常: {e}", "ERROR")
        return False, 0

def generate_all_volumes():
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
            log(f"卷 {vol_num} 生成失败，停止", "ERROR")
            return False
        if count != CHAPTERS_PER_VOLUME:
            log(f"卷 {vol_num} 生成章数不足：{count}/{CHAPTERS_PER_VOLUME}", "WARNING")
        time.sleep(2)
    final_outline = get_outline_detail()
    if final_outline:
        total_chapters = sum(len(v.get("chapters", [])) for v in final_outline.get("volumes", []))
        log(f"✅ 所有卷大纲生成完毕，总章节数: {total_chapters}")
        return True
    else:
        log("❌ 最终验证失败，大纲不存在", "ERROR")
        return False

# ========== 写作流程 API ==========
def get_progress():
    try:
        resp = requests.get(f"{API_BASE}/novel_id/{NOVEL_ID}/progress", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            vol = data.get("current_volume")
            ch = data.get("current_chapter")
            scene = data.get("current_scene")
            completed = data.get("chapter_completed", False)
            if vol is None:
                vol = 1
            if ch is None:
                ch = 1
            if scene is None:
                scene = 0
            return vol, ch, scene, completed
        elif resp.status_code == 404:
            return None, None, None, None
        else:
            log(f"查询进度失败: {resp.status_code}", "WARNING")
            return None, None, None, None
    except Exception as e:
        log(f"查询进度异常: {e}", "WARNING")
        return None, None, None, None

def wait_for_task(task_id, timeout=88000, poll_interval=30):
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

def chapter_file_exists(vol, ch):
    """检查章节生产文件是否存在且大小足够"""
    file_path = f"data/novels/{NOVEL_ID}/vol_{vol:03d}/chap_{ch:03d}.txt"
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        return True, size
    return False, 0

def count_scenes_in_chapter(vol, ch):
    """统计章节文件中的场景数（通过 scene break 标记）"""
    file_path = f"data/novels/{NOVEL_ID}/vol_{vol:03d}/chap_{ch:03d}.txt"
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 统计场景分隔符数量
        count = content.count("<!-- scene break -->")
        # 如果没有分隔符，至少有一个场景
        return count + 1 if count > 0 else (1 if len(content) > 100 else 0)
    except Exception:
        return 0

# ========== 主流程 ==========
def main():
    log("=" * 60)
    log(f"AI Factory 长篇小说生成脚本（{TOTAL_VOLUMES}卷 × {CHAPTERS_PER_VOLUME}章 = {TOTAL_VOLUMES * CHAPTERS_PER_VOLUME}章）")
    log(f"Novel ID: {NOVEL_ID}")
    log("🔄 自动修复模式已启用")
    log("=" * 60)

    if not acquire_lock():
        sys.exit(1)

    try:
        # ========== 第一步：生成完整大纲 ==========
        log("检查大纲完整性...")
        if not generate_all_volumes():
            log("大纲生成失败，无法继续", "ERROR")
            sys.exit(1)
        log("✅ 大纲已完整，开始写作...")

        # ========== 第二步：持续写作循环 ==========
        consecutive_failures = 0
        max_consecutive_failures = 5

        while True:
            vol, ch, scene, completed = get_progress()
            if vol is None:
                log("无法获取进度，退出", "ERROR")
                break

            if is_novel_completed(vol, ch):
                log("🎉 所有500章已生成完毕！🎉")
                break

            # 检查章节文件是否存在（仅用于诊断，不做进度判断）
            exists, size = chapter_file_exists(vol, ch)
            if exists:
                scene_count = count_scenes_in_chapter(vol, ch)
                log(f"📄 章节 {vol}:{ch} 文件已存在 (size={size} 字节, scenes={scene_count})")

            # ============================================================
            # 进度完全由 DB 的 writing_progress 决定。
            #
            # 重要：scene_completion.py 在章节完成时会把 current_chapter 设为 chapter+1，
            #       并设置 chapter_completed=True。因此 completed=True 时 current_chapter
            #       已经指向下一章了，不需要（也不能）force_advance。
            #
            # 脚本只做一件事：轮询触发 API resume。不做任何猜测性推进。
            # ============================================================
            if completed:
                log(f"ℹ️ DB chapter_completed=True (当前指针={vol}:{ch})，继续走 resume")
            # ============================================================

            log(f"🚀 开始生成第{vol}卷第{ch}章...")
            payload = {
                "user_input": "继续写作",
                "task_type": "scene_plan",
                "novel_id": NOVEL_ID,
                "resume": True
            }
            success, task_id = async_request("resume", payload)
            if not success or not task_id:
                log("启动写作任务失败", "ERROR")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log("连续失败次数过多，退出", "ERROR")
                    break
                time.sleep(10)
                continue

            success, _ = wait_for_task(task_id, timeout=72000)
            if not success:
                log(f"第{vol}卷第{ch}章生成失败", "ERROR")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log("连续失败次数过多，退出", "ERROR")
                    break
                time.sleep(10)
                continue

            # 重置失败计数
            consecutive_failures = 0
            log(f"✅ 第{vol}卷第{ch}章完成")

            # 每10章输出一次熵值
            if (ch % 10) == 0:
                entropy = fetch_latest_entropy()
                if entropy:
                    log(f"📊 最新叙事熵 - local={entropy['local']:.3f}, arc={entropy['arc']:.3f}, civ={entropy['civ']:.3f}")

            # 短暂休息，避免API过载
            time.sleep(2)

        log("=" * 60)
        log("✅ 所有章节生成完成！")
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
