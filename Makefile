.PHONY: help setup lock test goldens bench bench-one overload serve clean tree
.DEFAULT_GOAL := help

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
ENGINE  ?= naive
WORKLOAD ?= mixed
CONC    ?= 1,4,16
PORT    ?= 8000

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

setup: $(VENV)/bin/activate  ## Create venv and install pinned deps
	$(PIP) install --upgrade pip pip-tools
	$(PIP) install -r requirements.txt

lock: $(VENV)/bin/activate  ## Recompile requirements.txt from requirements.in (NFR3)
	$(VENV)/bin/pip-compile --strip-extras --output-file=requirements.txt requirements.in

lock-gpu: $(VENV)/bin/activate  ## Recompile requirements-gpu.txt — LINUX/CUDA BOX ONLY
	@python3 -c "import sys; sys.exit(0 if sys.platform.startswith('linux') else 1)" || \
		{ echo "requirements-gpu.txt must be generated on the Linux GPU box (triton ships no macOS wheels). Run this on Day 12."; exit 1; }
	$(VENV)/bin/pip-compile --strip-extras --output-file=requirements-gpu.txt requirements-gpu.in

test:  ## M1 gate — run before every commit
	$(PY) -m pytest tests/ -v

goldens:  ## Regenerate M1 golden token IDs (deliberate act; see scripts/make_goldens.py)
	$(PY) -m scripts.make_goldens

bench:  ## Sweep every implemented engine (the M5 target)
	$(PY) -m inference_server.bench.run --engine all --workload $(WORKLOAD) --concurrency $(CONC)

bench-one:  ## Bench one engine: make bench-one ENGINE=naive
	$(PY) -m inference_server.bench.run --engine $(ENGINE) --workload $(WORKLOAD) --concurrency $(CONC)

overload:  ## P4 (Day 11): 30 min at 10x capacity, expects a running server (M4)
	$(PY) -m inference_server.bench.run --engine $(ENGINE) --workload overload --concurrency 320 --tag overload

serve:  ## Run the HTTP server: make serve ENGINE=naive
	ENGINE=$(ENGINE) $(PY) -m uvicorn inference_server.server.app:app --port $(PORT)

tree:  ## Show the layout (what an interviewer sees first)
	@find . -name '*.py' -o -name 'Makefile' -o -name '*.md' | grep -v '$(VENV)' | sort

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
