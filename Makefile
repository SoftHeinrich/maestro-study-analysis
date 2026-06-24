.PHONY: analyze test clean
analyze:
	python analyze.py --threshold 3
	python analyze.py --threshold 4
test:
	python tests/test_reproduce.py
clean:
	rm -rf outputs/*.md outputs/*.csv __pycache__ studies/__pycache__ tests/__pycache__
