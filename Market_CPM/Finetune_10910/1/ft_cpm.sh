#!/usr/bin/env sh
set -e

/home/lizhuowan/caffe_cpm/build/tools/caffe train \
    --solver=/home/lizhuowan/Market_CPM/Finetune_10910/1/pose_solver.prototxt \
	--weights=/home/lizhuowan/Market_CPM/Finetune_10910/1/caffemodel/pose2_iter_6000.caffemodel \
	--gpu=0 \
	$@
