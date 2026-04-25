#!/usr/bin/env python
"""Serial long-cycle test: 50 sequential API calls to verify stability."""

import asyncio
import aiohttp
import time

async def call_api(session, i, user_input):
    url = "http://localhost:8000/api/v1/execute"
    payload = {"user_input": user_input}
    start = time.time()
    try:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            elapsed = time.time() - start
            success = result.get("success", False)
            code_executed = result.get("code_executed", False)
            print(f"[{i:02d}] {elapsed:.2f}s | success={success} | code_executed={code_executed}")
            return {"index": i, "success": success, "elapsed": elapsed}
    except Exception as e:
        print(f"[{i:02d}] ERROR: {e}")
        return {"index": i, "success": False, "error": str(e)}

async def main():
    user_input = "写一个Python函数计算斐波那契数列第n项，并测试n=10"
    total = 50
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(total):
            result = await call_api(session, i+1, user_input)
            results.append(result)
            await asyncio.sleep(0.5)  # 串行间隔，降低负载

    success_count = sum(1 for r in results if r.get("success"))
    total_time = sum(r.get("elapsed", 0) for r in results)
    print("\n=== Summary ===")
    print(f"Total requests: {total}")
    print(f"Success: {success_count}")
    print(f"Failed: {total - success_count}")
    print(f"Average time: {total_time/total:.2f}s")
    print(f"Total time: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())