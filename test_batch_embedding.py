#!/usr/bin/env python
import asyncio
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.writing.summarizer import generate_embedding, batch_generate_embeddings, cosine_similarity
from src.config import config

async def test_single():
    print("=== 测试单个 embedding ===")
    text = "测试文本"
    emb_str = await generate_embedding(text)
    emb = json.loads(emb_str)
    print(f"长度: {len(emb)}, 前5: {emb[:5]}")

async def test_batch_short():
    print("\n=== 测试批量短文本 (2个) ===")
    texts = ["测试文本1", "测试文本2"]
    embeddings = await batch_generate_embeddings(texts)
    print(f"返回 {len(embeddings)} 个向量")
    for i, emb in enumerate(embeddings):
        print(f"  文本{i}: 长度={len(emb)}, 前5={emb[:5]}")

async def test_batch_with_long():
    print("\n=== 测试批量包含长文本 (模拟 scene_text 2000字符) ===")
    long_text = "修仙小说正文 " * 500  # 约3500字符
    long_text = long_text[:2000]
    texts = [long_text, "主角飞升成仙", "捡到玉佩"]
    embeddings = await batch_generate_embeddings(texts)
    print(f"返回 {len(embeddings)} 个向量")
    for i, emb in enumerate(embeddings):
        print(f"  文本{i}: 长度={len(emb)}, 前5={emb[:5]}")

async def test_similarity():
    print("\n=== 测试相似度计算 ===")
    text1 = "主角飞升成仙"
    text2 = "主角突然飞升成仙"
    emb1_str = await generate_embedding(text1)
    emb2_str = await generate_embedding(text2)
    emb1 = json.loads(emb1_str)
    emb2 = json.loads(emb2_str)
    sim = cosine_similarity(emb1, emb2)
    print(f"相似度: {sim:.4f}")

async def test_realistic():
    print("\n=== 模拟真实验证场景 ===")
    scene_text = "林逸贴着藏书阁石壁挪动脚步，烛火在青铜机关兽上投下摇晃阴影。他摸到第三层书架时，腰间玉佩突然发烫，暗格里『咔』的轻响惊得他缩回手。守卫的脚步声由远及近，林逸屏息蹲伏在《玄天秘录》古籍堆后。青甲士兵腰牌闪过寒光，靴底摩擦声在密闭空间格外清晰。『哼，偏不信这机关打不通』，他盯着书脊凸起的云纹，指尖按压三下。暗门开启的瞬间，守卫正好转身，林逸翻进夹层时险些撞上对方胸甲。『有意思...』他压低声音咒骂，摸到石壁内嵌的青铜机括。当玉佩嵌入凹槽的刹那，整面书墙向两侧分开，露出幽深通道。「发现家族藏书阁密室」的兴奋让心跳轰鸣，林逸回头确认守卫仍未察觉，抓起地上的《太乙真经》残卷塞进怀中。"  # 实际正文
    must_events = ["发现家族藏书阁密室"]
    goal = "探索家族藏书阁寻找修炼线索"
    conflict = "藏书阁禁地机关复杂且有守卫巡逻"

    texts_to_embed = [scene_text[:2000]] + must_events + [goal, conflict]
    print(f"共 {len(texts_to_embed)} 个文本需要 embedding")
    start = time.time()
    try:
        embeddings = await batch_generate_embeddings(texts_to_embed)
        elapsed = time.time() - start
        print(f"批量请求耗时: {elapsed:.2f}s")
        scene_emb = embeddings[0]
        # 检查 must_events
        for i, evt in enumerate(must_events, start=1):
            evt_emb = embeddings[i]
            sim = cosine_similarity(evt_emb, scene_emb)
            print(f"must_event '{evt[:20]}...' 相似度: {sim:.4f}")
        # goal
        goal_emb = embeddings[-2]
        sim_goal = cosine_similarity(goal_emb, scene_emb)
        print(f"goal 相似度: {sim_goal:.4f}")
        # conflict
        conflict_emb = embeddings[-1]
        sim_conflict = cosine_similarity(conflict_emb, scene_emb)
        print(f"conflict 相似度: {sim_conflict:.4f}")
    except Exception as e:
        print(f"批量请求失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_single()
    await test_batch_short()
    await test_batch_with_long()
    await test_similarity()
    await test_realistic()

if __name__ == "__main__":
    asyncio.run(main())