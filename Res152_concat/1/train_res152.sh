#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res152_concat/1/solver1.prototxt \
	--weights=/home/lizhuowan/Market1501/Res152_pos30_regression/2/model/Res152_1_iter_130000.caffemodel \
	--gpu=0 \
	$@
