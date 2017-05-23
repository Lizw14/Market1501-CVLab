#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res50_concat/regression/1/solver1.prototxt \
	--weights=/home/lizhuowan/caffe/examples/Res50On1501/ResNet-50-model.caffemodel \
	--gpu=0 \
	$@
