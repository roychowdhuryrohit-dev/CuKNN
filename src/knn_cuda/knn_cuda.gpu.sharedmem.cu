#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define K 3
#define IMAGE_SIZE 784
#define NUM_TRAIN 50000
#define NUM_VAL 10000
#define BLOCK_SIZE 256
#define BATCH_SIZE 100  // Process 100 validation images at once
#define NUM_CLASSES 10

typedef struct {
    float distance;
    int label;
} DistanceLabel;

// Function to read MNIST image file
void readMNISTImages(const char* filename, float* images, int num_images, int offset) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // Read and verify magic number
    int magic_number = 0;
    size_t read_items = fread(&magic_number, sizeof(int), 1, fp);
    if (read_items != 1) {
        printf("Error reading magic number\n");
        fclose(fp);
        exit(1);
    }

    // Convert from big-endian to little-endian
    magic_number = ((magic_number & 0xff000000) >> 24) |
                  ((magic_number & 0x00ff0000) >> 8) |
                  ((magic_number & 0x0000ff00) << 8) |
                  ((magic_number & 0x000000ff) << 24);

    if (magic_number != 2051) {
        printf("Invalid magic number: %d\n", magic_number);
        fclose(fp);
        exit(1);
    }

    // Skip the rest of the header
    fseek(fp, 12, SEEK_CUR);

    // Seek to offset if needed
    if (offset > 0) {
        fseek(fp, offset * IMAGE_SIZE, SEEK_CUR);
    }

    // Read image data
    unsigned char* temp = (unsigned char*)malloc(num_images * IMAGE_SIZE);
    if (!temp) {
        printf("Failed to allocate temporary buffer\n");
        fclose(fp);
        exit(1);
    }

    size_t read_size = fread(temp, 1, num_images * IMAGE_SIZE, fp);
    printf("Read %zu bytes from images file\n", read_size);

    if (read_size != num_images * IMAGE_SIZE) {
        printf("Error: Expected to read %d bytes but got %zu\n",
               num_images * IMAGE_SIZE, read_size);
        free(temp);
        fclose(fp);
        exit(1);
    }

    // Convert to float and normalize
    for (int i = 0; i < num_images * IMAGE_SIZE; i++) {
        images[i] = (float)temp[i] / 255.0f;
    }

    free(temp);
    fclose(fp);
}

// Function to read MNIST label file
void readMNISTLabels(const char* filename, int* labels, int num_labels, int offset) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // Read and verify magic number
    int magic_number = 0;
    size_t read_items = fread(&magic_number, sizeof(int), 1, fp);
    if (read_items != 1) {
        printf("Error reading magic number from labels file\n");
        fclose(fp);
        exit(1);
    }

    // Convert from big-endian to little-endian
    magic_number = ((magic_number & 0xff000000) >> 24) |
                  ((magic_number & 0x00ff0000) >> 8) |
                  ((magic_number & 0x0000ff00) << 8) |
                  ((magic_number & 0x000000ff) << 24);

    if (magic_number != 2049) {
        printf("Invalid magic number in labels file: %d\n", magic_number);
        fclose(fp);
        exit(1);
    }

    // Skip number of items
    fseek(fp, 4, SEEK_CUR);

    // Seek to offset if needed
    if (offset > 0) {
        fseek(fp, offset, SEEK_CUR);
    }

    // Read labels
    unsigned char* temp = (unsigned char*)malloc(num_labels);
    if (!temp) {
        printf("Failed to allocate temporary buffer for labels\n");
        fclose(fp);
        exit(1);
    }

    size_t read_size = fread(temp, 1, num_labels, fp);
    printf("Read %zu bytes from labels file\n", read_size);

    if (read_size != num_labels) {
        printf("Error: Expected to read %d bytes but got %zu\n",
               num_labels, read_size);
        free(temp);
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < num_labels; i++) {
        labels[i] = (int)temp[i];
    }

    free(temp);
    fclose(fp);
}

// CUDA kernel using shared memory
__global__ void computeDistances(float* train_images,
                               float* val_image,
                               DistanceLabel* distances,
                               int num_train) {
    __shared__ float shared_val_image[IMAGE_SIZE];

    // Load validation image into shared memory
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Collaborative loading of validation image into shared memory
    for (int i = tid; i < IMAGE_SIZE; i += blockDim.x) {
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

// CUDA kernel for finding k nearest neighbors
__global__ void findKNearest(DistanceLabel* distances,
                           int* labels,
                           int num_train,
                           int k,
                           int* predicted_label) {
    __shared__ int label_counts[NUM_CLASSES];
    __shared__ float min_distances[K];
    __shared__ int min_labels[K];

    int tid = threadIdx.x;

    // Initialize shared memory
    if (tid < NUM_CLASSES) {
        label_counts[tid] = 0;
    }
    if (tid < K) {
        min_distances[tid] = INFINITY;
        min_labels[tid] = -1;
    }
    __syncthreads();

    // Each thread processes its portion of distances
    for (int i = tid; i < num_train; i += blockDim.x) {
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

    // Find most common label
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
	printf("Usage: %s <train_images_file_path> <train_labels_file_path>\n", argv[0]);
	printf("<train_images_file_path>: Path to the train-images-idx3-ubyte\n");
	printf("<train_labels_file_path>: Path to the train-labels-idx1-ubyte\n");
	exit(1);
    }

    const char* images_filepath = argv[1];
    const char* labels_filepath = argv[2];

    printf("Starting KNN MNIST Classification with Shared Memory...\n");
    clock_t total_start = clock();

    // Allocate and load data (same as before)...
    printf("Allocating host memory...\n");
    float* h_train_images = (float*)malloc(NUM_TRAIN * IMAGE_SIZE * sizeof(float));
    float* h_val_images = (float*)malloc(NUM_VAL * IMAGE_SIZE * sizeof(float));
    int* h_train_labels = (int*)malloc(NUM_TRAIN * sizeof(int));
    int* h_val_labels = (int*)malloc(NUM_VAL * sizeof(int));

    if (!h_train_images || !h_val_images || !h_train_labels || !h_val_labels) {
        printf("Error: Host memory allocation failed\n");
        return 1;
    }

    // Read data
    printf("Reading training data...\n");
    readMNISTImages(images_filepath, h_train_images, NUM_TRAIN, 0);
    readMNISTLabels(labels_filepath, h_train_labels, NUM_TRAIN, 0);

    printf("Reading validation data...\n");
    readMNISTImages(images_filepath, h_val_images, NUM_VAL, NUM_TRAIN);
    readMNISTLabels(labels_filepath, h_val_labels, NUM_VAL, NUM_TRAIN);

    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_train_images, *d_val_image;
    int *d_train_labels, *d_predicted_label;
    DistanceLabel *d_distances;

    cudaMalloc(&d_train_images, NUM_TRAIN * IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_val_image, IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_train_labels, NUM_TRAIN * sizeof(int));
    cudaMalloc(&d_predicted_label, sizeof(int));
    cudaMalloc(&d_distances, NUM_TRAIN * sizeof(DistanceLabel));

    // Copy training data to device
    printf("Copying training data to device...\n");
    cudaMemcpy(d_train_images, h_train_images,
               NUM_TRAIN * IMAGE_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, h_train_labels,
               NUM_TRAIN * sizeof(int),
               cudaMemcpyHostToDevice);

    // Process validation set
    int correct_predictions = 0;
    printf("\nProcessing validation set...\n");
    clock_t batch_start = clock();

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((NUM_TRAIN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < NUM_VAL; i++) {
        // Copy current validation image to device
        cudaMemcpy(d_val_image, &h_val_images[i * IMAGE_SIZE],
                  IMAGE_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice);

        // Compute distances using shared memory
        computeDistances<<<gridDim, blockDim>>>(d_train_images, d_val_image,
                                              d_distances, NUM_TRAIN);

        // Find k nearest neighbors
        findKNearest<<<1, BLOCK_SIZE>>>(d_distances, d_train_labels,
                                      NUM_TRAIN, K, d_predicted_label);

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
                   i + 1, NUM_VAL,
                   (float)correct_predictions / (i + 1) * 100,
                   batch_time, total_time);

            batch_start = clock();
        }
    }

    // Print final results
    clock_t end = clock();
    double total_time = (double)(end - total_start) / CLOCKS_PER_SEC;
    float accuracy = (float)correct_predictions / NUM_VAL * 100;

    printf("\nFinal Results:\n");
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
