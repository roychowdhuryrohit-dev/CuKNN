# CuKNN
This project is about performance optimisation by running experiments on different stages of image recognition using KNN classification and MNIST dataset.

## Project Structure
- **/data**: This folder contains the MNIST dataset in binary format.
- **/bin**: This folder contains all the binaries generated.
- **/lib**: This folder contains the important dependencies from the following C++ libraries.
  - *Eigen*: For linear algebra operations.
  - *nlohmann/json*: For JSON building/parsing.
  - *stb*: For image loading/decoding.
- **/src**: This folder contains all the source files for preprocessing and training/inference.
- **/reports**: This folder contains some reports collected during experiments.


## Build 
To build the binaries make sure the following tools and frameworks are installed.
- g++
- OpenMP
- nvcc
- python3 (numpy, matplotlib)

Simply run

```sh
make
cd data && python3 generate_images.py
```

## Run

1. To run the preprocessing steps.
```sh
touch data/training_data.json data/testing_data.json
#Serial implementation, PCA components = 50
bin/run_pca_serial data/training_data 50 data/training_data.json #training set
bin/run_pca_serial data/testing_data 50 data/testing_data.json #testing set
#OMP implementation, PCA components = 50
bin/run_pca_omp data/training_data 50 data/training_data.json #training set
bin/run_pca_omp data/testing_data 50 data/testing_data.json #testing set
```

2. To run the training/testing steps.
```sh
#Run training/testing with PCA components
bin/knn_cuda_pca data/training_data.json data/testing_data.json
#Run training/testing with raw images
bin/knn_cuda_cpu_openmp data/train-images.idx3-ubyte data/train-labels.idx1-ubyte 8 #OpenMP, number of threads = 8
bin/knn_cuda_cpu_serials data/train-images.idx3-ubyte data/train-labels.idx1-ubyte #Serial
bin/knn_cuda_gpu_standard data/train-images.idx3-ubyte data/train-labels.idx1-ubyte #GPU, standard
bin/knn_cuda_gpu_2dblock data/train-images.idx3-ubyte data/train-labels.idx1-ubyte #GPU, 2D block
bin/knn_cuda_gpu_sharedmem data/train-images.idx3-ubyte data/train-labels.idx1-ubyte #GPU, shared memory
```

