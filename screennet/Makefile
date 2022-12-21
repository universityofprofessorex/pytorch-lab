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

pytype:
	pytype *

py-spy:
	py-spy record -o profile.svg -- python main.py

py-spy-top:
	py-spy top -- python main.py

autoflake:
# autoflake --recursive --remove-unused-variables --remove-all-unused-imports --in-place *.py
	autoflake --recursive --in-place *.py

pynguin:
	mkdir -p /tmp/pynguin-results || true
	pynguin \
        --project-path ./ \
        --output-path /tmp/pynguin-results \
        --module-name main \
        -v
