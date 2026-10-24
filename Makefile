.PHONY: install test demo data eval train
install:
	pip install -e ".[dev,app]"
test:
	pytest -q
data:
	python -m sdd.cli generate --out data/synthetic --n-good 40 --n-bad 160
train: data
	python -m sdd.cli train data/synthetic --out models/classifier.joblib
eval: data
	python -m sdd.cli evaluate data/synthetic --out docs/EVAL_RULES.md
demo:
	python -m sdd.cli inspect examples --out outputs
