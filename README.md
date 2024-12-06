# CuKNN: Performance Optimization using KNN Classification and MNIST Dataset

This project focuses on performance optimization by running experiments on
different stages of image recognition using KNN classification and the
popular MNIST dataset.

## Project Structure

Our project is organized into the following directories:

- **/data**: This folder contains the MNIST dataset in binary format.
- **/bin**: This folder holds all the binaries generated during the
experiment.
- **/lib**: This directory contains essential dependencies from various
C++ libraries, including:
  - *Eigen*: For efficient linear algebra operations.
  - *nlohmann/json*: For building and parsing JSON data.
  - *stb*: For loading and decoding images.
- **/src**: This folder houses the source files for preprocessing and
training/inference.
- **/reports**: This directory contains some reports collected during the
experiments.

## Build

To build the binaries, ensure that the following tools and frameworks are
installed:

* g++
* OpenMP
* nvcc (for CUDA support)
* python3 (with numpy and matplotlib)

Run the following commands to generate the binaries and dataset:

```sh
make
cd data && python3 generate_images.py
```

## Run

### Preprocessing Steps

```sh
touch data/training_data.json data/testing_data.json
# Serial implementation, PCA components = 50
bin/run_pca_serial data/training_data 50 data/training_data.json #training set
bin/run_pca_serial data/testing_data 50 data/testing_data.json #testing set
# OMP implementation, PCA components = 50
bin/run_pca_omp data/training_data 50 data/training_data.json #training set
bin/run_pca_omp data/testing_data 50 data/testing_data.json #testing set
```

### Training/Testing Steps

```sh
#Run training/testing with PCA components
bin/knn_cuda_pca data/training_data.json data/testing_data.json
# Run training/testing with raw images
bin/knn_cuda_cpu_openmp data/train-images.idx3-ubyte data/train-labels.idx1-ubyte 8 # OpenMP, number of threads = 8
bin/knn_cuda_cpu_serials data/train-images.idx3-ubyte data/train-labels.idx1-ubyte # Serial
bin/knn_cuda_gpu_standard data/train-images.idx3-ubyte data/train-labels.idx1-ubyte # GPU, standard
bin/knn_cuda_gpu_2dblock data/train-images.idx3-ubyte data/train-labels.idx1-ubyte # GPU, 2D block
bin/knn_cuda_gpu_sharedmem data/train-images.idx3-ubyte data/train-labels.idx1-ubyte # GPU, shared memory
```
