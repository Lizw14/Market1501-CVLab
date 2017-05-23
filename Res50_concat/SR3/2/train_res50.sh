#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res50_concat/SR3/2/solver1.prototxt \
	--weights=/home/lizhuowan/Market1501/Res50_concat/Init/Init_2branch_Imagenet.caffemodel \
	--gpu=0 \
	$@
