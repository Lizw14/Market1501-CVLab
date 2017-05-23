#!/usr/bin/env sh
set -e

/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/build/tools/caffe train \
    --solver=/home/lizhuowan/Market1501/Res152_concat/3/solver1.prototxt \
	--snapshot=/home/lizhuowan/Market1501/Res152_concat/3/model/Res152_1_iter_90000.solverstate \
	--gpu=0 \
	$@
