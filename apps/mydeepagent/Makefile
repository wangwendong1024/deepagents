.PHONY: sync test lint format type
sync:
	uv sync --group test
test:
	uv run --no-sync pytest
lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .
	uv run --no-sync ty check src
format:
	uv run --no-sync ruff check --fix .
	uv run --no-sync ruff format .
type:
	uv run --no-sync ty check src
