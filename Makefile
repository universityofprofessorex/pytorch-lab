link-conda-env:
	ln -sf environments-and-requirements/environment-mac.yml environment.yml

conda-update:
	conda env update
	pip freeze > installed.txt

conda-activate:
	pyenv activate anaconda3-2022.05
	conda activate pytorch-lab3

