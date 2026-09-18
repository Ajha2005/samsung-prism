# PRISM Theme 1 — one-command entry points.
.PHONY: help install install-eval test demo demo-versioned ablation eval docker clean

help:
	@echo "Targets:"
	@echo "  install       Install core deps (offline pipeline, demo, tests)"
	@echo "  install-eval  Install competition deps (mteb + sentence-transformers + CPU torch)"
	@echo "  test          Run the full offline test suite"
	@echo "  demo          Retrieval demo on real queries (offline backend)"
	@echo "  demo-versioned  Cross-version / evolutionary demo"
	@echo "  ablation      Ablation sweep on the offline synthetic set"
	@echo "  eval          Real CoIR AppsRetrieval eval -> results/appsretrieval_results.json"
	@echo "  docker        Build the reproducible Docker image"

install:
	pip install -r requirements.txt && pip install -e .

install-eval:
	pip install torch --index-url https://download.pytorch.org/whl/cpu \
	  && pip install -r requirements-eval.txt && pip install -e .

test:
	python -m pytest -q

demo:
	python -m prism.cli demo --backend hashing

demo-versioned:
	python -m prism.cli demo --backend hashing --versioned

ablation:
	python -m prism.cli ablation --backend hashing

# The leaderboard run. Needs network (model + dataset download on first use).
eval:
	python -m prism.cli eval --output results/appsretrieval_results.json

docker:
	docker build -t prism-code-search .

clean:
	rm -rf .prism_cache runs results mteb_results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
