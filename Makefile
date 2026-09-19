.PHONY: check format test-all

check:
	ruff check src tests scripts
	pytest -m "not slow"

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

test-all:
	pytest
