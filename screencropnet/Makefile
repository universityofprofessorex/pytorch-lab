CURRENT_DIR := $(shell pwd)

train:
	python main.py --seed 42 --to-disk --debug --interactive --epochs 40

transfer_learning:
	python main.py --seed 42 --to-disk --debug --interactive --epochs 40 --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth

train_one:
	python main.py --seed 42 --debug --epochs 1 --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth

transfer_train: transfer_learning

evaluate:
	python main.py --to-disk --debug --test --epochs 5 --weights ./models/ScreenCropNetV1_ObjLocModelV1_TransferLearningpascalVOC_40_epochs.pth

test: evaluate

pylint:
	pylint --enable F,E --output-format=colorized --disable=all --rcfile ../lint-configs-python/python/pylintrc main.py

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
	python main.py --predict "~/Downloads/IMG_6400.PNG" --weights ./models/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth --to-disk

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

get-best-model:
	curl -L 'https://www.dropbox.com/s/m5mvmlv28x7xdty/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth?dl=1' > collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth
