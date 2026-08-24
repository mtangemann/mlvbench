.PHONY: ruff
ruff:
	uv run ruff check .

.PHONY: fix
fix:
	uv run ruff check --fix .

.PHONY: test
test:
	uv run pytest --cov=mlvbench tests

.PHONY: docs
docs:
	zensical serve

.PHONY: todos
todos:
	@rg --no-heading -n \
          --colors 'path:fg:cyan' \
          --colors 'line:fg:green' \
          --colors 'match:fg:black' \
          --colors 'match:style:nobold' \
          --glob "*.py" \
          '^\s*#\s*((TODO|IDEA|NOTE)\b:?.*)' \
          --replace '$$1' .
