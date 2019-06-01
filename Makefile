DIST_DIR ?= ${PWD}/dist
BUILD_DIR ?= ${PWD}/build
TAG ?= dcase20194_docker

TASK ?= 4
CONFIG_FN ?= src/config_exp.py
CONFIG_DEST_FN ?= src/config.py
JUPYTER_PORT ?= 8893
DISABLE_MP ?= 0
PYARGS ?=
EXP_TAG = $(shell grep exp_tag ${CONFIG_FN} | cut -d= -f2 | head -n1 | sed "s/[ \'\"]//g")

.PHONY: clean
clean:
	@find stored_data -type f -name "*epoch*" -delete

.PHONY: container
container:
	@rm -rf build_directory
	@mkdir build_directory
	@cp Makefile Dockerfile requirements.txt build_directory/
	@cd build_directory && docker build \
		--build-arg group=$(shell id -g -n ${USER}) \
		--build-arg user=${USER} \
		--build-arg user_id=$(shell id -u) \
		--build-arg group_id=$(shell id -g) \
		-t ${TAG} \
		.
	@rm -rf build_directory

.PHONY: train
train:
	@cp ${CONFIG_FN} ${CONFIG_DEST_FN}
	@mkdir -p stored_data/${EXP_TAG}
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		python3 ${PYARGS} src/main.py 2>&1 | tee -a stored_data/${EXP_TAG}/log.txt
	@rm -f ${CONFIG_DEST_FN}

.PHONY: train_small
train_small:
	@cp ${CONFIG_FN} ${CONFIG_DEST_FN}
	@mkdir -p stored_data/${EXP_TAG}
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		python3 ${PYARGS} src/main.py -s 1000 2>&1 | tee -a stored_data/${EXP_TAG}/log.txt
	@rm -f ${CONFIG_DEST_FN}

.PHONY: eval
eval:
	@cp ${CONFIG_FN} ${CONFIG_DEST_FN}
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		./make_preds stored_data/${EXP_TAG}/MeanTeacher_with_synthetic/model/_best ${EXP_TAG}
	@rm -f ${CONFIG_DEST_FN}


.PHONY: predict
predict:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		python ${PYARGS} main.py -c ${CONFIG_FN} -m predict

.PHONY: jup
jup:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-p ${JUPYTER_PORT}:${JUPYTER_PORT} \
		--rm \
		-ti \
		${TAG} \
		jupyter-lab --allow-root --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser 2>&1 | tee log.txt

.PHONY: extract
extract:
	nvidia-docker run \
		-e DISABLE_MP=${DISABLE_MP} \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		python feature_extraction/extract.py data/all_wavs.lst features.lst -c ${CONFIG_FN}
# python feature_extraction/extract.py one.lst /tmp/done.lst -c ${CONFIG_FN}

.PHONY: train_scaler
train_scaler:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		--rm \
		-ti \
		${TAG} \
		python feature_extraction/train_scaler.py \
			/corpora/DCASE2019_task4/dataset/metadata/train/features.lst \
			scaler.lfb.pkl \
			-c ${CONFIG_FN}

.PHONY: debug
debug:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-e DISABLE_MP=1 \
		-ti \
		--rm \
		${TAG} \
		bash
