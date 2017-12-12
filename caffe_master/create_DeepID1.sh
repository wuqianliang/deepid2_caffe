#!/usr/bin/env sh
# This script converts the CASIA-maxpy-clean into lmdb format.
set -e

EXAMPLE=examples/deepid1
DATA=data/ms-celeb-1m/MsCelebV1-Faces-Aligned-Parsed/ 
TOOLS=build/tools

RESIZE_HEIGHT=55
RESIZE_WIDTH=47

echo "creating lmdb..."

rm -rf $EXAMPLE/DeepID1_train_lmdb

rm -rf $EXAMPLE/DeepID1_test_lmdb


$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA \
    $DATA/train.txt \
    $EXAMPLE/DeepID1_train_lmdb

$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA \
    $DATA/val.txt \
    $EXAMPLE/DeepID1_test_lmdb

echo "compute image mean..."

$TOOLS/compute_image_mean $EXAMPLE/DeepID1_train_lmdb \
  $EXAMPLE/DeepID1_mean.proto

echo "done..."

