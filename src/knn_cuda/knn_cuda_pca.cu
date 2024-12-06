#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <knn_cuda_pca_lib.h>
#include <string>

// Define block dimensions
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define NUM_CLASSES 10
#define IMAGE_SIZE 500
#define K 3

typedef struct {
    float distance;
    int label;
} DistanceLabel;

// Modified compute distances kernel for 2D blocks
__global__ void computeDistances(float* train_images,
	float* val_image,
	DistanceLabel* distances,
	int num_train) {
    __shared__ float shared_val_image[IMAGE_SIZE];

    // 2D thread indexing
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidy * blockDim.x + tidx;  // Linear thread ID within block

    // 2D block indexing
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = gidy * gridDim.x * blockDim.x + gidx;  // Global linear ID

    // Collaborative loading of validation image
    int threads_per_block = blockDim.x * blockDim.y;
    for (int i = tid; i < IMAGE_SIZE; i += threads_per_block) {
	if (i < IMAGE_SIZE) {
	    shared_val_image[i] = val_image[i];
	}
    }
    __syncthreads();

    if (gid < num_train) {
	float sum = 0.0f;
#pragma unroll 32
	for (int i = 0; i < IMAGE_SIZE; i++) {
	    float diff = train_images[gid * IMAGE_SIZE + i] - shared_val_image[i];
	    sum += diff * diff;
	}
	distances[gid].distance = sqrtf(sum);
    }
}

// Modified k-nearest neighbors kernel for 2D blocks
__global__ void findKNearest(DistanceLabel* distances,
	int* labels,
	int num_train,
	int* predicted_label) {
    __shared__ int label_counts[NUM_CLASSES];
    __shared__ float min_distances[K];
    __shared__ int min_labels[K];

    // Convert 2D thread index to linear index
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = tidy * blockDim.x + tidx;

    // Initialize shared memory
    if (tid < NUM_CLASSES) {
	label_counts[tid] = 0;
    }
    if (tid < K) {
	min_distances[tid] = INFINITY;
	min_labels[tid] = -1;
    }
    __syncthreads();

    // Calculate stride for thread processing
    int threads_per_block = blockDim.x * blockDim.y;

    // Each thread processes its portion of distances
    for (int i = tid; i < num_train; i += threads_per_block) {
	float curr_dist = distances[i].distance;
	int curr_label = labels[i];

	// Insert into sorted array of K smallest distances
	for (int j = 0; j < K; j++) {
	    if (curr_dist < min_distances[j]) {
		// Shift larger distances
		for (int l = K-1; l > j; l--) {
		    min_distances[l] = min_distances[l-1];
		    min_labels[l] = min_labels[l-1];
		}
		min_distances[j] = curr_dist;
		min_labels[j] = curr_label;
		break;
	    }
	}
    }
    __syncthreads();

    // Count labels of K nearest neighbors
    if (tid < K) {
	atomicAdd(&label_counts[min_labels[tid]], 1);
    }
    __syncthreads();

    // Thread 0 finds the most common label
    if (tid == 0) {
	int max_count = 0;
	int max_label = 0;
	for (int i = 0; i < NUM_CLASSES; i++) {
	    if (label_counts[i] > max_count) {
		max_count = label_counts[i];
		max_label = i;
	    }
	}
	*predicted_label = max_label;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
	printf("Usage: %s <training_json_file_path> <testing_json_file_path>\n", argv[0]);
	printf("<training_json_file_path>: Path to the json file containing images and labels for training\n");
	printf("<testing_json_file_path>: Path to the json file containing images and labels for testing\n");
	exit(1);
    }

    std::string training_filepath = argv[1];
    std::string testing_filepath = argv[2];

    printf("Starting KNN MNIST Classification with 2D Blocks (%dx%d)...\n",
	    BLOCK_DIM_X, BLOCK_DIM_Y);
    clock_t total_start = clock();

    float* h_train_images;
    float* h_val_images;
    int* h_train_labels;
    int* h_val_labels;

    int image_size;
    int num_train;
    int num_val;


    // Read JSON data
    printf("Reading training JSON data...\n");
    loadJSON(training_filepath, h_train_images, h_train_labels, image_size, num_train);

    printf("Reading testing data...\n");
    loadJSON(testing_filepath, h_val_images, h_val_labels, image_size, num_val);
    
    printf("Image Size - %d Training Set Size = %d Testing Set Size = %d\n", image_size, num_train, num_val);
    if (!h_train_images || !h_val_images || !h_train_labels || !h_val_labels) {
	printf("Error: Host memory allocation failed\n");
	return 1;
    }
    /*printf("First 10 image values of train\n");
    for (int i=0;i<10;i++){
	printf("%f\n", h_train_images[i]);
    }
    printf("First label of train\n");
    printf("%d\n", h_train_labels[0]);*/
    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_train_images, *d_val_image;
    int *d_train_labels, *d_predicted_label;
    DistanceLabel *d_distances;

    cudaMalloc(&d_train_images, num_train * IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_val_image, IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_train_labels, num_train * sizeof(int));
    cudaMalloc(&d_predicted_label, sizeof(int));
    cudaMalloc(&d_distances, num_train * sizeof(DistanceLabel));

    // Copy training data to device
    printf("Copying training data to device...\n");
    cudaMemcpy(d_train_images, h_train_images,
	    num_train * IMAGE_SIZE * sizeof(float),
	    cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, h_train_labels,
	    num_train * sizeof(int),
	    cudaMemcpyHostToDevice);

    // Calculate grid dimensions properly
    int numBlocksX = (int)ceil(sqrt((float)num_train / (BLOCK_DIM_X * BLOCK_DIM_Y)));
    int numBlocksY = numBlocksX;  // Using square grid for simplicity

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim(numBlocksX, numBlocksY);

    printf("Using grid size: %dx%d blocks, each block %dx%d threads\n",
	    numBlocksX, numBlocksY, BLOCK_DIM_X, BLOCK_DIM_Y);

    // Set up 2D kernel configurations
    //dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    //dim3 gridDim(
    //    (int)ceil((float)NUM_TRAIN / (BLOCK_DIM_X * gridDim.x)),
    //    (int)ceil((float)NUM_TRAIN / (BLOCK_DIM_Y * gridDim.y))
    //);

    // Process validation set
    int correct_predictions = 0;
    printf("\nProcessing validation set...\n");
    clock_t batch_start = clock();

    for (int i = 0; i < num_val; i++) {
	// Copy current validation image to device
	cudaMemcpy(d_val_image, &h_val_images[i * IMAGE_SIZE],
		IMAGE_SIZE * sizeof(float),
		cudaMemcpyHostToDevice);

	// Compute distances using 2D blocks
	computeDistances<<<gridDim, blockDim>>>(d_train_images, d_val_image,
		d_distances, num_train);

	// Find k nearest neighbors
	dim3 knn_block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
	findKNearest<<<1, knn_block_dim>>>(d_distances, d_train_labels,
		num_train, d_predicted_label);

	// Get prediction
	int predicted_label;
	cudaMemcpy(&predicted_label, d_predicted_label,
		sizeof(int), cudaMemcpyDeviceToHost);

	if (predicted_label == h_val_labels[i]) {
	    correct_predictions++;
	}

	// Print progress every 1000 images
	if ((i + 1) % 1000 == 0) {
	    clock_t current = clock();
	    double batch_time = (double)(current - batch_start) / CLOCKS_PER_SEC;
	    double total_time = (double)(current - total_start) / CLOCKS_PER_SEC;

	    printf("Processed %d/%d images (%.2f%% accuracy so far, batch time: %.3fs, total time: %.3fs)\n",
		    i + 1, num_val,
		    (float)correct_predictions / (i + 1) * 100,
		    batch_time, total_time);

	    batch_start = clock();
	}
    }

    // Print final results
    clock_t end = clock();
    double total_time = (double)(end - total_start) / CLOCKS_PER_SEC;
    float accuracy = (float)correct_predictions / num_val * 100;

    printf("\nFinal Results:\n");
    printf("Block size: %dx%d\n", BLOCK_DIM_X, BLOCK_DIM_Y);
    printf("Accuracy: %.2f%%\n", accuracy);
    printf("Total execution time: %.3f seconds\n", total_time);

    // Cleanup
    cudaFree(d_train_images);
    cudaFree(d_val_image);
    cudaFree(d_train_labels);
    cudaFree(d_predicted_label);
    cudaFree(d_distances);
    free(h_train_images);
    free(h_val_images);
    free(h_train_labels);
    free(h_val_labels);

    return 0;
}
