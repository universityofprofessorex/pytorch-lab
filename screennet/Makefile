train:
	python -i main.py

pylint:
	pylint --enable F,E --rcfile ../lint-configs-python/python/pylintrc main.py

mypy:
	mypy --follow-imports=silent --ignore-missing-imports --show-column-numbers --no-pretty main.py

black:
	black *.py

isort:
	isort --atomic .
