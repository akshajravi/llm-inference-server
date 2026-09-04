.PHONY: help setup lock lock-gpu test test-light goldens bench bench-one bench-http overload overload-smoke charts headline serve clean tree
.DEFAULT_GOAL := help

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
ENGINE  ?= naive
WORKLOAD ?= mixed
CONC    ?= 1,4,8,16,32
PORT    ?= 8000
#: Base URL of an already-running server for the HTTP targets (`make serve` in another
#: shell). 127.0.0.1 rather than localhost: on macOS the latter tries ::1 first.
URL     ?= http://127.0.0.1:$(PORT)
#: Offered load for `make overload`, in requests/s. "10x capacity" means 10x the req/s
#: the engine cleared at its best concurrency in the latest mixed sweep — `make headline`
#: prints that figure and the exact command. No default on purpose: it depends on the box.
RPS     ?=
DURATION ?= 1800

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

test-light:  ## Same suite with a 256-block pool (~290 MiB instead of 2.25 GiB per engine) for the dev Mac
	NUM_BLOCKS=256 $(PY) -m pytest tests/ -q

goldens:  ## Regenerate M1 golden token IDs (deliberate act; see scripts/make_goldens.py)
	$(PY) -m scripts.make_goldens

bench:  ## M5: sweep every implemented engine on `mixed` at CONC, write results/ (one command)
	$(PY) -m inference_server.bench.run --engine all --workload $(WORKLOAD) --concurrency $(CONC)

bench-one:  ## Bench one engine in-process: make bench-one ENGINE=naive
	$(PY) -m inference_server.bench.run --engine $(ENGINE) --workload $(WORKLOAD) --concurrency $(CONC)

bench-http:  ## Same closed-loop sweep over the wire against a running server at URL
	$(PY) -m inference_server.bench.run --http $(URL) --workload $(WORKLOAD) --concurrency $(CONC) --tag http

overload:  ## M4: open-loop 30 min at RPS req/s against a running server: make overload RPS=<10x capacity>
	@test -n "$(RPS)" || { echo "set RPS to 10x capacity — run 'make headline' for the figure, e.g. make overload RPS=30"; exit 1; }
	$(PY) -m inference_server.bench.run --http $(URL) --workload overload --rps $(RPS) --duration $(DURATION) --tag overload

overload-smoke:  ## 60 s open-loop at 20 req/s against URL, written to results/scratch (not committed)
	$(PY) -m inference_server.bench.run --http $(URL) --workload smoke-overload --out-dir results/scratch --tag smoke

charts:  ## Render docs/charts/*.png from the newest results files: make charts N=3
	$(PY) -m scripts.plot --latest $(or $(N),1)

headline:  ## Print M2/M3 ratios and the suggested overload RPS from the newest mixed sweep
	$(PY) -m scripts.headline $(FILES)

serve:  ## Run the HTTP server: make serve ENGINE=naive
	ENGINE=$(ENGINE) $(PY) -m uvicorn inference_server.server.app:app --port $(PORT)

tree:  ## Show the layout (what an interviewer sees first)
	@find . -name '*.py' -o -name 'Makefile' -o -name '*.md' | grep -v '$(VENV)' | sort

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
