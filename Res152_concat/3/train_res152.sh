#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res152_concat/3/solver1.prototxt \
	--weights=/home/lizhuowan/Market1501/Res152_concat/Init/Init_Imagenet.caffemodel \
	--gpu=0 \
	$@
