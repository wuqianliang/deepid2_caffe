#!/usr/bin/env sh
# This script converts the CASIA-maxpy-clean into lmdb format.
set -e

EXAMPLE=examples/deepid2
DATA=data/lfw-aligned/
TOOLS=build/tools

RESIZE_HEIGHT=96
RESIZE_WIDTH=96

echo "creating lmdb..."

rm -rf $EXAMPLE/DeepID2_train_lmdb

rm -rf $EXAMPLE/DeepID2_test_lmdb


$TOOLS/convert_imageset --shuffle \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA \
    $DATA/train.txt \
    $EXAMPLE/DeepID2_train_lmdb

$TOOLS/convert_imageset --shuffle \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA \
    $DATA/val.txt \
    $EXAMPLE/DeepID2_test_lmdb

echo "compute image mean..."

$TOOLS/compute_image_mean -backend=lmdb $EXAMPLE/DeepID2_train_lmdb \
  $EXAMPLE/DeepID2_mean.proto

echo "done..."

