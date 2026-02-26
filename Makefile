lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check .

lint-fix:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest .

check: lint test
