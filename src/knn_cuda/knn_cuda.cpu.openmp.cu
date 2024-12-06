#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <omp.h>
#include <stddef.h>
#include <cuda.h>

#define K 3
#define IMAGE_SIZE 784
#define NUM_TRAIN 50000
#define NUM_VAL 10000
#define BLOCK_SIZE 256

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

    // Skip the rest of the header (number of images, rows, columns)
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

// CUDA kernel for computing distances
__global__ void computeDistances(float* train_images, float* val_image,
                               DistanceLabel* distances, int num_train) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_train) {
        float sum = 0.0f;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            float diff = train_images[idx * IMAGE_SIZE + i] - val_image[i];
            sum += diff * diff;
        }
        distances[idx].distance = sqrtf(sum);
    }
}

int compareDistances(const void* a, const void* b) {
    return ((DistanceLabel*)a)->distance > ((DistanceLabel*)b)->distance ? 1 : -1;
}

// OpenMP optimized prediction
int predictLabel(DistanceLabel* distances, int num_train) {
    // Sort distances (sequential)
    qsort(distances, num_train, sizeof(DistanceLabel), compareDistances);

    int label_counts[10] = {0};

    // Parallel counting of K nearest neighbors
    #pragma omp parallel
    {
        int local_counts[10] = {0};

        #pragma omp for nowait
        for (int i = 0; i < K; i++) {
            local_counts[distances[i].label]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < 10; i++) {
                label_counts[i] += local_counts[i];
            }
        }
    }

    // Find most common label
    int max_count = 0;
    int predicted_label = 0;

    for (int i = 0; i < 10; i++) {
        if (label_counts[i] > max_count) {
            max_count = label_counts[i];
            predicted_label = i;
        }
    }

    return predicted_label;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
	printf("Usage: %s <train_images_file_path> <train_labels_file_path> <num_threads_omp>\n", argv[0]);
	printf("<train_images_file_path>: Path to the train-images-idx3-ubyte\n");
	printf("<train_labels_file_path>: Path to the train-labels-idx1-ubyte\n");
	printf("<num_threads_omp>: Number of threads used to OpenMP for prediction\n");
	exit(1);
    }

    const char* images_filepath = argv[1];
    const char* labels_filepath = argv[2];


    printf("Starting KNN MNIST Classification...\n");

    // Set number of OpenMP threads
    int num_threads = atoi(argv[3]);  // Change this number to control thread count
    omp_set_num_threads(num_threads);
    printf("Using %d OpenMP threads for prediction\n", num_threads);

    // Allocate host memory
    printf("Allocating host memory...\n");
    float* h_train_images = (float*)malloc(NUM_TRAIN * IMAGE_SIZE * sizeof(float));
    float* h_val_images = (float*)malloc(NUM_VAL * IMAGE_SIZE * sizeof(float));
    int* h_train_labels = (int*)malloc(NUM_TRAIN * sizeof(int));
    int* h_val_labels = (int*)malloc(NUM_VAL * sizeof(int));

    if (!h_train_images || !h_val_images || !h_train_labels || !h_val_labels) {
        printf("Error: Host memory allocation failed\n");
        return 1;
    }

    // Read training data
    printf("Reading training data...\n");
    readMNISTImages(images_filepath, h_train_images, NUM_TRAIN, 0);
    readMNISTLabels(labels_filepath, h_train_labels, NUM_TRAIN, 0);

    // Read validation data
    printf("Reading validation data...\n");
    readMNISTImages(images_filepath, h_val_images, NUM_VAL, NUM_TRAIN);
    readMNISTLabels(labels_filepath, h_val_labels, NUM_VAL, NUM_TRAIN);

    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_train_images, *d_val_image;
    DistanceLabel *d_distances;

    cudaMalloc(&d_train_images, NUM_TRAIN * IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_val_image, IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_distances, NUM_TRAIN * sizeof(DistanceLabel));

    // Copy training data to device
    cudaMemcpy(d_train_images, h_train_images,
               NUM_TRAIN * IMAGE_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    // Allocate host memory for distances
    DistanceLabel* h_distances = (DistanceLabel*)malloc(NUM_TRAIN * sizeof(DistanceLabel));

    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((NUM_TRAIN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Process validation set
    int correct_predictions = 0;
    printf("\nProcessing validation set...\n");
    double batch_start = omp_get_wtime();
    double validation_start = omp_get_wtime();

    for (int i = 0; i < NUM_VAL; i++) {
        // Copy current validation image to device
        cudaMemcpy(d_val_image, &h_val_images[i * IMAGE_SIZE],
                   IMAGE_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Compute distances
        computeDistances<<<gridDim, blockDim>>>(d_train_images, d_val_image,
                                              d_distances, NUM_TRAIN);

        // Copy distances back to host
        cudaMemcpy(h_distances, d_distances,
                   NUM_TRAIN * sizeof(DistanceLabel),
                   cudaMemcpyDeviceToHost);

        // Assign labels in parallel
        #pragma omp parallel for
        for (int j = 0; j < NUM_TRAIN; j++) {
            h_distances[j].label = h_train_labels[j];
        }

        // Predict label
        int predicted_label = predictLabel(h_distances, NUM_TRAIN);
        if (predicted_label == h_val_labels[i]) {
            correct_predictions++;
        }

        // Print progress every 1000 images
        if ((i + 1) % 1000 == 0) {
            double current = omp_get_wtime();
            double batch_time = current - batch_start;
            double total_time = current - validation_start;

            printf("Processed %d/%d images (%.2f%% accuracy so far, batch time: %.3fs, total time: %.3fs)\n",
                   i + 1, NUM_VAL,
                   (float)correct_predictions / (i + 1) * 100,
                   batch_time, total_time);

            batch_start = current;
        }
    }

    // Print final results
    double end = omp_get_wtime();
    double total_time = end - validation_start;
    float accuracy = (float)correct_predictions / NUM_VAL * 100;

    printf("\nValidation Results:\n");
    printf("Correct predictions: %d/%d\n", correct_predictions, NUM_VAL);
    printf("Accuracy: %.2f%%\n", accuracy);
    printf("Total execution time: %.3f seconds\n", total_time);

    // Cleanup
    free(h_train_images);
    free(h_val_images);
    free(h_train_labels);
    free(h_val_labels);
    free(h_distances);
    cudaFree(d_train_images);
    cudaFree(d_val_image);
    cudaFree(d_distances);

    return 0;
}
