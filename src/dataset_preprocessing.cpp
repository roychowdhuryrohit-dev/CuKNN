#include "dataset_preprocessing.h"
#include <Eigen/Dense>
#include <filesystem>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fs = std::filesystem;


std::vector<ImageData> load_images_from_directory(const std::string& directory) {
    std::vector<ImageData> images;
    std::vector<std::string> image_paths;

    // Collect all image paths first then process them in parallel afterwards
    for (const auto& label_entry : fs::directory_iterator(directory)) {
        if (label_entry.is_directory()) {
            std::string label = label_entry.path().filename().string();
            // For each label directory, collect the image files
            for (const auto& img_entry : fs::directory_iterator(label_entry.path())) {
                if (img_entry.is_regular_file() && img_entry.path().extension() == ".png") {  // Assuming PNG image files
                    image_paths.push_back(img_entry.path().string());
                }
            }
        }
    }

    // Parallelize the image loading and processing
    #pragma omp parallel
    {
        std::vector<ImageData> local_images;
        #pragma omp for
        for (size_t i = 0; i < image_paths.size(); ++i) {
            const std::string& image_path = image_paths[i];
            int width, height, channels;
            unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
            if (!data) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                continue;
            }
            if (width != 28 || height != 28) {
                std::cerr << "Image " << image_path << " is not 28x28, skipping." << std::endl;
                stbi_image_free(data);
                continue;
            }

            // Flatten the image into a 1D vector and then save
            std::vector<float> flattened_image(data, data + (width * height * channels));
            stbi_image_free(data);
            std::string label = fs::path(image_path).parent_path().filename().string();
            ImageData image_data = { flattened_image, label };
            local_images.push_back(image_data);
        }

        // Merge local results into the global vector
        #pragma omp critical
        {
            images.insert(images.end(), local_images.begin(), local_images.end());
        }
    }

    return images;
}


void normalize_images(std::vector<std::vector<float>>& images) {
    #pragma omp parallel for
    for (auto& image : images) {
        for (auto& pixel : image) {
            pixel /= 255.0f;  // Normalize image data to [0, 1] range
        }
    }
}

void apply_pca(const Eigen::MatrixXf& data, int n_components, Eigen::MatrixXf& reduced_data) {
    Eigen::MatrixXf centered_data = data.rowwise() - data.colwise().mean();
    Eigen::MatrixXf covariance_matrix = (centered_data.transpose() * centered_data) / float(data.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance_matrix);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Eigen decomposition failed!" << std::endl;
        return;
    }
    // Get the top 'n_components' eigenvectors
    Eigen::MatrixXf eigenvectors = solver.eigenvectors().rightCols(n_components);
    reduced_data = centered_data * eigenvectors;
}

void preprocess_images_with_pca(std::vector<ImageData>& images, int n_components) {
    int n_samples = images.size();
    int n_pixels = images[0].image.size();

    std::vector<std::vector<float>> flattened_images(n_samples, std::vector<float>(n_pixels));
    for (int i = 0; i < n_samples; ++i) {
        flattened_images[i] = images[i].image;
    }

    normalize_images(flattened_images);

    Eigen::MatrixXf data(n_samples, n_pixels);
    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_pixels; ++j) {
            data(i, j) = flattened_images[i][j];
        }
    }

    Eigen::MatrixXf reduced_data;
    apply_pca(data, n_components, reduced_data);

    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> reduced_vector;
        for (int j = 0; j < n_components; ++j) {
            reduced_vector.push_back(reduced_data(i, j));
        }
        images[i].image = reduced_vector;
    }
}

