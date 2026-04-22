.PHONY: install test run docker-up docker-down clean lint format

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up -d postgres

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-full:
	docker-compose up --build -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/

lint:
	black --check src/ tests/

format:
	black src/ tests/

migrate-db:
	bash scripts/init_db.sh
