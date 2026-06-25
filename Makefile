.PHONY: analyze impute figures test clean venv
analyze:
	python analyze.py --threshold 3
	python analyze.py --threshold 4
# Pilot skip-unrated vs TF-IDF-imputed GT (~1 min first run, then cached).
impute:
	python analyze.py --threshold 3 --impute
# Figures need matplotlib; use the venv (see `make venv`).
figures:
	. .venv/bin/activate && python figures.py --threshold 3
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install matplotlib
test:
	python tests/test_reproduce.py
clean:
	rm -rf outputs/*.md outputs/*.csv outputs/figures __pycache__ studies/__pycache__ tests/__pycache__
