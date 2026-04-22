#!/usr/bin/env python
"""Initialize PostgreSQL database with pgvector extension and tables."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import psycopg2
from src.config.settings import config

def init_database():
    """Create tables for knowledge base."""
    conn = psycopg2.connect(config.postgres_dsn)
    conn.autocommit = True
    cur = conn.cursor()
    
    # 1. Enable pgvector extension
    print("Creating vector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # 2. Create chunks table
    print("Creating chunks table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255),
            content TEXT NOT NULL,
            embedding vector(768),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # 3. Create HNSW index
    print("Creating HNSW index...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
        ON chunks 
        USING hnsw (embedding vector_cosine_ops)
    """)
    
    # 4. Create documents table
    print("Creating documents table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR(255) PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    cur.close()
    conn.close()
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    init_database()
