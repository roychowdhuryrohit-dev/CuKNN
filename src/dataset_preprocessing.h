#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>


struct ImageData {
    std::vector<float> image;
    std::string label;
};

std::vector<ImageData> load_images_from_directory(const std::string& directory);

void normalize_images(std::vector<std::vector<float>>& images);

void apply_pca(const Eigen::MatrixXf& data, int n_components, Eigen::MatrixXf& reduced_data);

void preprocess_images_with_pca(std::vector<ImageData>& images, int n_components);

