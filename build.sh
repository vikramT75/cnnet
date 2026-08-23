#!/bin/sh

set -xe

# source ../raylib.env && ./build.sh
# source ../raylib.env && ./build.sh && ./demos/adder
# source ../raylib.env && ./build.sh && ./upscalenn ./mnist/training0/3.png
# source ../raylib.env && ./build.sh && ./morph2nn ./mnist/training0/3.png ./mnist/training0/7.png
# source ../raylib.env && ./build.sh && ./demos/train <model.arch> <model.mat>
CFLAGS="-O3 -march=native -ffast-math `pkg-config --cflags raylib`"
#CFLAGS2="-O3 -march=native -ffast-math"
LIBS="`pkg-config --libs raylib` -lm -lX11 -ldl -pthread"
#LIBS2="-lm"
CC="gcc"
$CC $CFLAGS -o demos/train demos/train.c $LIBS 
#$CC $CFLAGS -o adder_gen adder_gen.c $LIBS 
$CC $CFLAGS -o ./demos/adder ./demos/adder.c $LIBS
#$CC $CFLAGS -o demos/xor_gen demos/xor_gen.c $LIBS 
$CC $CFLAGS -o ./demos/xor ./demos/xor.c $LIBS
#$CC -O3 -o dump_nn dump_nn.c -lm
#$CC -o dump_nn dump_nn.c -lm
#$CC $CFLAGS -o upscalenn upscalenn.c $LIBS
#$CC $CFLAGS -o morph2nn morph2nn.c $LIBS

# shape_gen
$CC -O3 -march=native -ffast-math -o demos/shape_gen demos/shape_gen.c -lm
# mnist_gen
$CC -O3 -march=native -ffast-math -o demos/mnist_gen demos/mnist_gen.c -lm
# Usage:
#   ./demos/shape_gen
#   ./demos/mnist_gen mnist/training/ demos/data/mnist_train.mat
#   ./demos/mnist_gen mnist/testing/  demos/data/mnist_test.mat
#   ./demos/infer demos/models/mnist.arch demos/data/mnist.arch.weights.mat demos/models/mnist.labels
#   ./demos/infer demos/models/shapes.arch demos/data/shapes.arch.weights.mat demos/models/shapes.labels
$CC -O3 -march=native -ffast-math -o demos/trainer demos/trainer.c -lm
$CC $CFLAGS -o demos/infer demos/infer.c $LIBS
