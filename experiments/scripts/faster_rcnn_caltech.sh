#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$0


TRAIN_IMDB="caltech_train"
TEST_IMDB="caltech_test"
PT_DIR="caltech"
ITERS=1000



time ./tools/train_net.py --gpu ${GPU_ID} \
  --weights  data/pretrain_model/VGG_imagenet.npy
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_caltech.yml \
  ${EXTRA_ARGS}

