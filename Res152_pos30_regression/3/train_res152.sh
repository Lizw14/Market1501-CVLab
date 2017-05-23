#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res152_pos30_regression/3/solver1.prototxt \
	--weights=/home/lizhuowan/caffe/examples/Res152On1501/ResNet-152-model.caffemodel \
	--gpu=0 \
	$@
