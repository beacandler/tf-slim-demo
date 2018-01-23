#!/usr/bin/env bash

_DATA_URL='http://download.tensorflow.org/example_images/flower_photos.tgz'
DATASET_DIR='/tmp/flower_photos/images'

rm -rf ${DATASET_DIR}
if [ ! -d "$DATASET_DIR" ]; then
    mkdir -p ${DATASET_DIR}
fi

if [ ! -f ${DATASET_DIR}/flower_photos.tgz ]; then
    wget ${_DATA_URL}
    tar -xf flower_photos.tgz
    mv flower_photos/* ${DATASET_DIR}
    rm -rf flower_photos
    rm flower_photos.tgz
fi

CKPT_DIR='/tf-slim-demo/src/models'
rm -rf ${CKPT_DIR}
if [ ! -d "$CKPT_DIR" ]; then
    mkdir -p ${CKPT_DIR}
fi

if [ ! -f ${CKPT_DIR}/inception_v3.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvf inception_v3_2016_08_28.tar.gz
    mv inception_v3.ckpt ${CKPT_DIR}/inception_v3.ckpt
    rm inception_v3_2016_08_28.tar.gz
fi
