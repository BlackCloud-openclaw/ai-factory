#!/usr/bin/env python3
"""
压力测试脚本 - 支持按时间持续运行
用法: python stress_test.py --concurrency 5 --duration 18000   # 18000秒=5小时
"""

import asyncio
import aiohttp
import time
import argparse
import random
from typing import List, Dict, Any
from collections import defaultdict

TASKS = [
    {"name": "code_fib", "input": "写一个Python函数计算斐波那契数列第n项，并测试n=10"},
    {"name": "code_sort", "input": "写一个Python函数实现快速排序，并测试"},
    {"name": "writing_rain", "input": "写一段小说开头，描述一个雨夜的场景"},
    {"name": "writing_morning", "input": "写一段小说开头，描述一个清晨的街景"},
]

DEFAULT_URL = "http://localhost:8000/api/v1/execute"
DEFAULT_CONCURRENCY = 5
DEFAULT_DURATION = 3600  # 默认1小时
TIMEOUT = 1200  # 单个请求超时秒数

class Stats:
    def __init__(self):
        self.success = 0
        self.failed = 0
        self.times = []
        self.errors = defaultdict(int)
        self.lock = asyncio.Lock()
        self.start_time = None
        self.end_time = None

    async def record_success(self, duration: float):
        async with self.lock:
            self.success += 1
            self.times.append(duration)

    async def record_failure(self, error_msg: str):
        async with self.lock:
            self.failed += 1
            self.errors[error_msg] += 1

    def report(self):
        total_time = self.end_time - self.start_time
        print("\n" + "="*50)
        print("压力测试报告")
        print("="*50)
        print(f"测试总耗时: {total_time:.2f}s ({total_time/3600:.2f}小时)")
        print(f"总请求数: {self.success + self.failed}")
        print(f"成功: {self.success}")
        print(f"失败: {self.failed}")
        print(f"成功率: {self.success/(self.success+self.failed)*100:.2f}%")
        if self.times:
            sorted_times = sorted(self.times)
            avg = sum(self.times)/len(self.times)
            p50 = sorted_times[int(len(sorted_times)*0.5)]
            p95 = sorted_times[int(len(sorted_times)*0.95)]
            p99 = sorted_times[int(len(sorted_times)*0.99)]
            print(f"平均响应时间: {avg:.2f}s")
            print(f"P50: {p50:.2f}s")
            print(f"P95: {p95:.2f}s")
            print(f"P99: {p99:.2f}s")
        print(f"吞吐量: {(self.success+self.failed)/total_time:.2f} 请求/秒")
        if self.errors:
            print("\n错误分布:")
            for err, count in self.errors.items():
                print(f"  {err}: {count}次")

async def send_request(session: aiohttp.ClientSession, url: str, task_input: str, stats: Stats):
    start = time.time()
    try:
        async with session.post(url, json={"user_input": task_input}, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("success", False):
                    duration = time.time() - start
                    await stats.record_success(duration)
                else:
                    error = data.get("error", "Unknown error")
                    await stats.record_failure(f"API returned success=false: {error}")
            else:
                text = await resp.text()
                await stats.record_failure(f"HTTP {resp.status}: {text[:50]}")
    except asyncio.TimeoutError:
        await stats.record_failure("Timeout")
    except Exception as e:
        await stats.record_failure(f"Exception: {type(e).__name__}: {str(e)}")

async def worker(session: aiohttp.ClientSession, url: str, task_queue: asyncio.Queue, stats: Stats, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            # 从队列获取任务（如果队列为空且未停止，则等待）
            task_input = await asyncio.wait_for(task_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if task_input is None:
            break
        await send_request(session, url, task_input, stats)
        task_queue.task_done()

async def main(concurrency: int, duration_seconds: int, url: str):
    print(f"压力测试开始: 并发数={concurrency}, 持续时间={duration_seconds}s ({duration_seconds/3600:.1f}小时), URL={url}")
    print("任务类型: 随机混合 (代码生成/小说写作)")

    stats = Stats()
    stats.start_time = time.time()

    # 任务队列，初始为空，由生产者动态填充
    task_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # 生产者：持续生成随机任务，直到时间结束
    async def producer():
        end_time = stats.start_time + duration_seconds
        while time.time() < end_time:
            task = random.choice(TASKS)
            await task_queue.put(task["input"])
            # 避免任务堆积过多，控制生产速率（可选）
            await asyncio.sleep(0.01)
        # 停止信号
        stop_event.set()
        # 向队列放入 None 唤醒等待的 worker
        for _ in range(concurrency):
            await task_queue.put(None)

    connector = aiohttp.TCPConnector(limit=concurrency*2, limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [asyncio.create_task(worker(session, url, task_queue, stats, stop_event)) for _ in range(concurrency)]
        producer_task = asyncio.create_task(producer())

        await producer_task
        await task_queue.join()   # 等待队列中所有任务被处理
        await asyncio.gather(*workers)

    stats.end_time = time.time()
    stats.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Factory API 压力测试（按时间持续）")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"并发数 (默认: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--duration", "-d", type=int, default=DEFAULT_DURATION,
                        help=f"测试持续时间（秒） (默认: {DEFAULT_DURATION}s，即1小时)")
    parser.add_argument("--url", "-u", type=str, default=DEFAULT_URL,
                        help=f"API 地址 (默认: {DEFAULT_URL})")
    args = parser.parse_args()

    asyncio.run(main(args.concurrency, args.duration, args.url))