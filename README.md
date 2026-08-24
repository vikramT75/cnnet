# Neural Network in C

A lightweight neural network framework built in C. The core library operates on a custom memory arena allocator and implements modern machine learning primitives, including the Adam optimiser, cross-entropy loss and batch processing.

The framework trains multi-class classifiers, like MNIST and includes an interactive GUI for real-time inference.

It also features visual Raylib demonstrations for modelling logic gates and learning continuous spatial mappings for image upscaling and morphing.

## Features

- **Core Library (`nn.h`):** A single-header library implementing forward propagation, backpropagation and **OpenMP**-accelerated vectorised batched matrix operations.
- **Modern ML Tooling:** Implements the Adam Optimiser and logarithmic Cross-Entropy Loss with Softmax activation.
- **Interactive Visualiser (`nn_ui.h`):** Raylib-based GUI rendering network architectures, live cost plots and drawing canvases.
- **Data Pipeline:** Uses text-based manifests to parse raw PNGs directly into binary `.mat` matrices.

## Programs

**Computer Vision & Implicit Networks**

- **Classifier (`classifier_train.c` & `classifier_infer.c`):** Multi-class training and an interactive Raylib drawing GUI, respectively, for real-time digit/shape prediction.
- **Upscaler (`upscalenn.c`):** A Neural Representation that learns to map `[x, y]` spatial coordinates to grayscale intensities, thus memorising and continuously upscaling an image.
- **Image Morphing (`morph2nn.c`):** Extends the spatial coordinate mapping to smoothly interpolate and morph between two distinct images in real-time.

**Logic & Fundamentals**

- **XOR Gate (`xor.c`):** Non-linear classification modelling with live architecture and cost visualisation.
- **Binary Adder (`adder.c`):** Multi-input, multi-output learning for bitwise addition with visualisation.

**Data & Utilities**

- **Dataset Generators (`mnist_gen.c` & `shape_gen.c`):** Utility programs that read text-based manifests and compile raw PNGs into the binary `.mat` format for training.

## Directory Structure

- `nn.h` & `nn_ui.h`: Core math, memory and visualisation libraries.
- `Makefile`: Multi-core build system configured for hardware-accelerated multi-threaded execution (`-march=native -ffast-math -fopenmp`).
- `demos/`: Source code for all executables.
  - `data/` & `models/`: Binary `.mat` matrices, manifests and `.arch` network configs.
- `build/`: Output directory for compiled executables.

## Getting Started

**Prerequisites:**

- GCC (with **OpenMP** support) and Raylib (`pkg-config --cflags raylib`).
- The raw [MNIST PNG dataset](https://www.kaggle.com/datasets/jidhumohan/mnist-png) to be extracted into the `mnist/` folder.

### Building & Cleaning

To compile all executables into the `build/` directory, run:

```bash
make -j
```

To remove all compiled binaries, run:

```bash
make clean
```

### Image Classification

To train and test a multi-class classifier, follow these three steps:

1. **Generate Dataset:**
   _(Requires the MNIST PNG dataset to be present in the `mnist/` directory)_
   ```bash
   make data-mnist
   ```
2. **Train the Model:**

   ```bash
   make train-mnist EPOCHS=50  # Default is 100
   ```

   _(Tip: To restrict OpenMP CPU thread usage, use `OMP_NUM_THREADS=N` , for example: `OMP_NUM_THREADS=4 make train-mnist`)_

3. **Run Live Inference:**
   ```bash
   make infer-mnist
   ```
   _(Note: Replace `mnist` with `shapes` in the commands above to work with the shapes dataset instead)._

### Other Demos

Launch the visual demos using the following commands:

```bash
make upscale
make morph
make xor
make adder
```

### Interactive UI Controls

**Classifier**

- `Left-Click` - Draw on canvas
- `Right-Click` - Erase
- `C` - Clear Canvas

**Other Demos**

- `Space` or `P` - Pause/Resume training visualisation
- `R` - Randomise network weights and restart
