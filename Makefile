.PHONY: *

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check .

lint-fix:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest .

test-coverage:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

check: lint test
