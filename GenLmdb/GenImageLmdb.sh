#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/lizhuowan/Market1501/lmdb/lmdb_12936/Image_lmdb_12936
DATA=/home/lizhuowan/Market1501/Anno_result/Image_Market_12936.txt
TOOLS=/home/lizhuowan/caffe/build/tools
MEAN=$EXAMPLE.binaryproto

TRAIN_DATA_ROOT=/home/lizhuowan/Market1501/Market-1501-v15.09.15/bounding_box_train/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_1501.sh to the path" \
       "where the Market1501 training data is stored."
  exit 1
fi

echo "Creating train lmdb..."

$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $TRAIN_DATA_ROOT \
    $DATA \
    $EXAMPLE
	
$TOOLS/compute_image_mean \
    $EXAMPLE \
    $MEAN
  
echo "Done."
