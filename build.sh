#!/bin/sh

set -xe

# source ../raylib.env && ./build.sh
# source ../raylib.env && ./build.sh && ./demos/adder
# source ../raylib.env && ./build.sh && ./upscalenn ./mnist/training/3.png
# source ../raylib.env && ./build.sh && ./morph2nn ./mnist/training/3.png ./mnist/training/7.png
# source ../raylib.env && ./build.sh && ./demos/train <model.arch> <model.mat>
CFLAGS="-O3 `pkg-config --cflags raylib`"
#CFLAGS2="-O3"
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
