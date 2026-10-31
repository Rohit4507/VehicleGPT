.PHONY: help check check-data model specs docs test test-data

help:
	@echo "make check      - docs + feasibility model + stdlib tests (no deps needed)"
	@echo "make check-data - everything, including the CAN/simulation test suite"
	@echo "make model      - print the feasibility model report"
	@echo "make specs      - print the canonical P0-P3 spec table (markdown)"
	@echo "make docs       - lint markdown against tools/paths.py"
	@echo "make test       - stdlib tests only (fast, dependency-free)"
	@echo "make test-data  - data-plane tests (needs cantools, python-can, numpy, PyYAML)"
	@echo "make demo       - run the CAN round-trip demo on a virtual bus"

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
