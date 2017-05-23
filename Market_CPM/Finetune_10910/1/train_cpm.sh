#!/usr/bin/env sh
set -e

/home/lizhuowan/caffe_cpm/build/tools/caffe train \
    --solver=/home/lizhuowan/Market_CPM/Finetune_10910/1/pose_solver.prototxt \
	--weights=/home/lizhuowan/convolutional-pose-machines-release/model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel \
	--gpu=1 \
	$@
