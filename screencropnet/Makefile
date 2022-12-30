CURRENT_DIR := $(shell pwd)

train:
	python main.py --to-disk --debug --interactive --epochs 10

pylint:
	pylint --enable F,E --rcfile ../lint-configs-python/python/pylintrc main.py

mypy:
	mypy --follow-imports=silent --ignore-missing-imports --show-column-numbers --no-pretty main.py

black:
	black .

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
memray:
	memray run --live main.py

tensorboard:
	tensorboard --logdir runs/

predict:
	python main.py --predict "~/Downloads/2020-11-25_10-47-32_867.jpeg" --weights ./models/ScreenNetV1.pth --interactive --to-disk

bad-predict:
	python main.py --predict ../bad_image_test.jpg --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth --interactive --to-disk

label-studio:
	docker run -it --rm -p 8080:8080 \
	-v $(CURRENT_DIR)/mydata:/label-studio/data \
	--env LOCAL_FILES_SERVING_ENABLED=true \
	--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	--env LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
	--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
	--env JSON_LOG=1 \
	--env LOG_LEVEL=DEBUG \
	-v $(CURRENT_DIR)/myfiles:/label-studio/files \
	heartexlabs/label-studio:latest \
	label-studio

worst-first:
	python main.py --to-disk --debug --predict ~/dev/bossjones/pytorch-lab/scratch/datasets/twitter_facebook_tiktok/test --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth --worst-first --results ./worst-first.csv

zip-dataset:
	bash ./zip-dataset.sh
	ls -ltah ~/Downloads/datasets/twitter_screenshots_localization_dataset.zip