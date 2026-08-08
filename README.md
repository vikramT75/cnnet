# Neural Network in C

A lightweight, zero-dependency neural network library implemented from scratch in C, featuring real-time training visualization with Raylib and an implicit neural representation image upscaler.

## Features

- Core neural network library with forward pass, analytical backpropagation, gradient descent, and finite difference gradient checking.
- Implicit neural representation upscaler (mapping normalized pixel coordinates to grayscale intensity).
- Real-time training visualizer showing loss curves and weight node connections.
- Validation demos including a 4-bit binary adder and an XOR gate.

## Directory Structure

- `nn.h`: Core single-header neural network library.
- `img2nn.c`: Application that trains a neural network on an image coordinate grid and outputs an upscaled version.
- `build.sh`: Bash script to compile the upscaler.
- `demos/`: Validation examples (XOR, binary adder, and generalized trainer).
- `thirdparty/`: External dependencies (Raylib headers, stb_image, stb_image_write, and sv.h).
- `mnist/`: Training image assets and evaluation outputs.

## Prerequisites

- GCC compiler.
- Raylib library installed on your system (development headers and binaries).

## Building and Running

To compile the image upscaling application:

```bash
chmod +x build.sh
./build.sh
```

To run the upscaler on a grayscale image:

```bash
./img2nn <path_to_grayscale_image.png>
```

Controls during visualization:
- Space or P: Pause/resume training.
- R: Reset training, randomize weights, and clear the cost plot.
