.PHONY: all data analysis test
all: data analysis test
data:
	python data/generate_synthetic.py
analysis:
	python analysis/run_analysis.py
test:
	python -m pytest -q
