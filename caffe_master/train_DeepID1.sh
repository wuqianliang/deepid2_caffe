#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/deepid1/deepID1_solver.prototxt $@
