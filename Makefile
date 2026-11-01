.PHONY: help check check-data model specs docs test test-data demo store ask storage trends episodes precursors emulate bakeoff bakeoff-required

help:
	@echo "make check      - docs + feasibility model + stdlib tests (no deps needed)"
	@echo "make check-data - everything, including the CAN/simulation test suite"
	@echo "make model      - print the feasibility model report"
	@echo "make specs      - print the canonical P0-P3 spec table (markdown)"
	@echo "make docs       - lint markdown against tools/paths.py"
	@echo "make test       - stdlib tests only (fast, dependency-free)"
	@echo "make test-data  - data-plane tests (needs cantools, python-can, numpy, PyYAML)"
	@echo "make demo       - run the CAN round-trip demo on a virtual bus"
	@echo "make store      - build the 14-day store from the simulated program"
	@echo "make ask        - answer the five brief questions from the store"
	@echo "make emulate    - calibrate this host against the Tier C envelope"
	@echo "make bakeoff    - model bake-off; set MODELS='a.gguf b.gguf'"

check: docs model test
	@echo "all checks passed"

check-data: check test-data
	@echo "all checks passed (including data plane)"

docs:
	@python3 tools/check_docs.py

model:
	@python3 tools/feasibility_model.py

specs:
	@python3 tools/feasibility_model.py --specs

# Dependency-free: only the consistency tests run here, so CI can check docs
# and the feasibility model on a bare Python install.
test:
	@python3 -m unittest discover -s tests -p "test_consistency.py"

test-data:
	@python3 -m pytest tests -q

demo:
	@python3 tools/phase0/demo_can_pipeline.py

# Throughput a candidate model must reach to meet the target, per path.
# Needs no model and no weights: pure arithmetic over tools/paths.py.
bakeoff-required:
	@python3 tools/phase0/bakeoff.py --required-only

# MODELS='~/models/a.gguf ~/models/b.gguf' make bakeoff
bakeoff:
	@python3 tools/phase0/bakeoff.py --models $(MODELS) --eval

# Deterministic layer: simulate 14 days, store it, answer the brief's questions.
DB ?= /tmp/vehiclegpt.db

store:
	@python3 tools/phase0/build_store.py --db $(DB) --days 14

ask: store
	@python3 tools/phase0/ask.py --db $(DB) --all

# The searchable episode cards, and a lexical search over them.
episodes: store
	@python3 tools/phase0/episodes.py --db $(DB) --list

# What moved before an event, by precedence and by lagged correlation.
precursors: store
	@python3 tools/phase0/precursors.py --db $(DB) --event warning --window 1500

# Where a series changed level, and which readings broke away from it.
trends: store
	@python3 tools/phase0/trends.py --db $(DB)

# Stress-test the store against the 103 MB budget, under both sampling assumptions.
storage: store
	@python3 tools/phase0/storage_report.py --db $(DB)

# Why a laptop pinned to two cores is still not a Tier C device.
emulate:
	@python3 tools/phase0/emulate.py --calibrate
