#!/usr/bin/env python
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from src.knowledge.retrieval import KnowledgeRetriever
from src.db import init_db_pool, close_db_pool
from src.writing.summarizer import generate_embedding

async def test_retrieval():
    # 初始化数据库池
    await init_db_pool()
    retriever = KnowledgeRetriever()
    
    # 1. 测试存储文档和分块
    doc_id = "test_doc_001"
    chunks = [
        {"content": "修仙小说主角林逸，从炼气期开始修炼。", "chunk_index": 0, "metadata": {"source": "test"}},
        {"content": "他在家族选拔赛中击败对手，进入外门。", "chunk_index": 1, "metadata": {"source": "test"}},
    ]
    # 生成向量
    embeddings = []
    for chunk in chunks:
        emb_str = await generate_embedding(chunk["content"])
        emb = json.loads(emb_str)
        embeddings.append(emb)
    await retriever.store_document(doc_id, "测试文档", "test.txt", "txt", "全文内容")
    await retriever.store_chunks(chunks, doc_id, embeddings)
    print("✅ 文档和分块存储成功")
    
    # 2. 测试关键词搜索
    results = await retriever.search("林逸", k=2)
    print(f"关键词搜索结果: {len(results)} 条")
    for r in results:
        print(f"  - {r['content'][:50]}... (score={r['score']:.3f})")
    
    # 3. 测试向量搜索
    query_emb_str = await generate_embedding("林逸修炼")
    query_emb = json.loads(query_emb_str)
    results_vec = await retriever.search_with_embedding(query_emb, k=2, threshold=0.3)
    print(f"向量搜索结果: {len(results_vec)} 条")
    for r in results_vec:
        print(f"  - {r['content'][:50]}... (score={r['score']:.3f})")
    
    await close_db_pool()
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_retrieval())