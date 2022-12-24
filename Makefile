link-conda-env:
	ln -sf environments-and-requirements/environment-mac.yml environment.yml

link-conda-env-intel:
	ln -sf environments-and-requirements/environment-mac-intel.yml environment.yml

conda-update:
	conda env update
	pip freeze > installed.txt

conda-activate:
	pyenv activate anaconda3-2022.05
	conda activate pytorch-lab3

conda-delete:
	conda env remove -n pytorch-lab3

conda-lock-env:
	conda env export > env.yml.lock
	conda list --explicit > spec-file.txt

conda-env-export:
	conda env export
	conda list --explicit

conda-history:
	conda env export --from-history

env-works:
	python ./contrib/is-mps-available.py
	python ./contrib/does-matplotlib-work.py

env-test: env-works

setup-dataset-scratch-env:
	bash contrib/setup-dataset-scratch-env.sh

download-dataset: setup-dataset-scratch-env
	curl -L 'https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1' > ./scratch/datasets/twitter_facebook_tiktok.zip
	unzip -l ./scratch/datasets/twitter_facebook_tiktok.zip

unzip-dataset:
	unzip ./scratch/datasets/twitter_facebook_tiktok.zip -d './scratch/datasets'
	rm -fv ./scratch/datasets/twitter_facebook_tiktok.zip

zip-dataset:
	bash contrib/zip-dataset.sh
	ls -ltah ./scratch/datasets/twitter_facebook_tiktok.zip

install-postgres:
	brew install postgresql@14

label-studio:
	label-studio
