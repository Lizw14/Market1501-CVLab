#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res50_concat/pos_init/1/solver1.prototxt \
	--weights=/home/lizhuowan/Market1501/Res50_concat/regression/1/model/Res50_1_iter_30000.caffemodel \
	--gpu=1 \
	$@
