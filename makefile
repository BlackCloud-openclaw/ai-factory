.PHONY: install test test-cov run docker-up docker-down docker-build docker-full clean lint format migrate-db model-up status help

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

docker-up:
	docker-compose up -d postgres

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-full:
	docker-compose up --build -d

model-up:
	@echo "Starting llama.cpp containers..."
	# 示例：启动 writing 模型
	docker run -d --rm --gpus all --name llamacpp-writing -p 8082:8081 \
		-v /models:/models ghcr.io/ggerganov/llama.cpp:server \
		-m /models/qwen3-32b-q5_k_m.gguf --host 0.0.0.0 --port 8081 -ngl 99
	@echo "Model container started. You may also start others."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/ logs/*.log

lint:
	black --check src/ tests/

format:
	black src/ tests/

migrate-db:
	python scripts/init_novel_db.py

status:
	@echo "=== Docker containers ==="
	docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo "\n=== Database tables ==="
	docker exec -it postgres psql -U woami -d ai_factory -c "\dt" 2>/dev/null || echo "PostgreSQL container not running"

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | awk -F':' '{print "  " $$1}'