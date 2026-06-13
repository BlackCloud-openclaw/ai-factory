.PHONY: install test test-cov run docker-up docker-down docker-build docker-full clean lint format migrate-db model-up status help

# 默认目标
.DEFAULT_GOAL := help

install:
	pip install -r requirements.txt

# 运行所有测试（不包括 e2e 和 characterization 的跳过标记）
test:
	pytest tests/causality/ tests/characterization/ -v --cov=src --cov-report=term-missing

# 完整测试（包括可能慢的测试）
test-all:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# 只运行因果层核心测试（快速）
test-core:
	pytest tests/causality/ -v -m "not e2e"

# 只运行确定性重放测试
test-replay:
	pytest tests/causality/test_deterministic_replay.py -v

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

# 使用环境变量文件运行
run-env:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env

# 启动LLM 容器（使用 docker-compose profile）
docker-models:
	docker compose --profile manual create
	docker start llamacpp-embedding postgres

model-down:
	docker compose --profile manual down

test-novel:
	python tests/simple_long_novel.py

# 数据库初始化（使用新脚本）
migrate-db:
	python scripts/init_novel_db.py

# 生成黄金数据集
generate-golden:
	python scripts/generate_golden.py --novel-id simple_long_novel_001

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/ logs/*.log

lint:
	black --check src/ tests/

format:
	black src/ tests/

# 检查数据库和模型状态
status:
	@echo "=== Docker containers ==="
	docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo "\n=== PostgreSQL tables ==="
	docker exec -it postgres psql -U woami -d ai_factory -c "\dt" 2>/dev/null || echo "PostgreSQL container not running"
	@echo "\n=== Projection health ==="
	docker exec -it postgres psql -U woami -d ai_factory -c "SELECT novel_id, projection_lag_events, drift_level, updated_at FROM projection_health;" 2>/dev/null || echo "Projection health table not found"

# 查看死信队列
dead-letters:
	docker exec -it postgres psql -U woami -d ai_factory -c "SELECT novel_id, event_id, error, retry_count, created_at FROM projection_dead_letters ORDER BY created_at DESC LIMIT 20;" 2>/dev/null || echo "No dead letters"

# 重置测试数据库
reset-test-db:
	-docker exec -it postgres psql -U woami -d postgres -c "DROP DATABASE IF EXISTS ai_factory_test WITH (FORCE);"
	-docker exec -it postgres psql -U woami -d postgres -c "CREATE DATABASE ai_factory_test OWNER woami;"
	@echo "Test database reset. Run 'make migrate-db' in test environment if needed."

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | awk -F':' '{print "  " $$1}'

.PHONY: audit
audit:
	python scripts/audit_replay.py $(NOVEL_ID)	

.PHONY: verify-projection
verify-projection:
	python scripts/verify_projection.py $(NOVEL_ID) --rebuild	