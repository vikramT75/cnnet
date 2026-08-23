export PKG_CONFIG_PATH := ../raylib/lib/pkgconfig/

CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra $(shell pkg-config --cflags raylib)
LIBS = $(shell pkg-config --libs raylib) -lm -lX11 -ldl -pthread

BUILD_DIR = build

TARGETS = $(BUILD_DIR)/classifier_train \
          $(BUILD_DIR)/classifier_infer \
          $(BUILD_DIR)/mnist_gen \
          $(BUILD_DIR)/shape_gen \
          $(BUILD_DIR)/morph2nn \
          $(BUILD_DIR)/upscalenn \
          $(BUILD_DIR)/adder \
          $(BUILD_DIR)/xor

.PHONY: all clean

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/classifier_train: demos/classifier_train.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD_DIR)/classifier_infer: demos/classifier_infer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/mnist_gen: demos/mnist_gen.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD_DIR)/shape_gen: demos/shape_gen.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD_DIR)/morph2nn: morph2nn.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/upscalenn: upscalenn.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/adder: demos/adder.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/xor: demos/xor.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(BUILD_DIR)/

.PHONY: data data-mnist data-shapes train-mnist train-shapes classifier_infer-mnist classifier_infer-shapes upscale morph run-xor run-adder


# Generate all datasets
data: data-mnist data-shapes

data-mnist: $(BUILD_DIR)/mnist_gen
	./$(BUILD_DIR)/mnist_gen demos/data/mnist_train_manifest.txt demos/data/mnist_train.mat
	./$(BUILD_DIR)/mnist_gen demos/data/mnist_test_manifest.txt demos/data/mnist_test.mat

data-shapes: $(BUILD_DIR)/shape_gen
	./$(BUILD_DIR)/shape_gen

# Default epochs for training
EPOCHS ?= 100
# parameterised usage : make train-mnist EPOCHS=50

run-xor: $(BUILD_DIR)/xor
	./$(BUILD_DIR)/xor

run-adder: $(BUILD_DIR)/adder
	./$(BUILD_DIR)/adder

upscale: $(BUILD_DIR)/upscalenn
	./$(BUILD_DIR)/upscalenn ./mnist/training0/3.png

morph: $(BUILD_DIR)/morph2nn
	./$(BUILD_DIR)/morph2nn ./mnist/training0/3.png ./mnist/training0/7.png

train-mnist: $(BUILD_DIR)/classifier_train
	./$(BUILD_DIR)/classifier_train demos/models/mnist.arch demos/data/mnist_train.mat demos/data/mnist_test.mat $(EPOCHS)

train-shapes: $(BUILD_DIR)/classifier_train
	./$(BUILD_DIR)/classifier_train demos/models/shapes.arch demos/data/shapes_train.mat demos/data/shapes_test.mat $(EPOCHS)

classifier_infer-mnist: $(BUILD_DIR)/classifier_infer
	./$(BUILD_DIR)/classifier_infer demos/models/mnist.arch demos/data/mnist.arch.weights.mat demos/models/mnist.labels

classifier_infer-shapes: $(BUILD_DIR)/classifier_infer
	./$(BUILD_DIR)/classifier_infer demos/models/shapes.arch demos/data/shapes.arch.weights.mat demos/models/shapes.labels
