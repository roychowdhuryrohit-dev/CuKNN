#include <iostream>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <chrono>
#include <cstdlib>

#include "dataset_preprocessing.h"

using json = nlohmann::json;


json image_data_to_json(const ImageData& image_data) {
    json j_obj;
    j_obj["label"] = image_data.label;
    j_obj["image"] = image_data.image;
    return j_obj;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <num_threads> <omp_schedule> <pca_ncomp> <output_file>" << std::endl;
        std::cerr << "  <image_directory>: Path to the directory containing images" << std::endl;
        std::cerr << "  <num_threads>: Number of OpenMP threads to use" << std::endl;
        std::cerr << "  <omp_schedule>: OpenMP scheduling policy (static, dynamic, guided)" << std::endl;
        std::cerr << "  <pca_ncomp>: Number of PCA components to use" << std::endl;
        std::cerr << "  <output_file>: Path to save the output JSON file" << std::endl;
        exit(1);
    }
    std::string image_directory = argv[1];
    int num_threads = std::stoi(argv[2]);
    std::string omp_schedule = argv[3];
    int pca_ncomp = std::stoi(argv[4]);
    std::string output_file = argv[5];

    // Set the number of OpenMP threads
    omp_set_num_threads(num_threads);

    // Set the OpenMP scheduling policy
    if (omp_schedule == "static") {
        omp_set_schedule(omp_sched_static, 0);
    } else if (omp_schedule == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, 0);
    } else if (omp_schedule == "guided") {
        omp_set_schedule(omp_sched_guided, 0);
    } else {
        std::cerr << "Invalid OpenMP scheduling policy. Defaulting to 'static'.\n";
        omp_set_schedule(omp_sched_static, 0);
    }

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Load images from the directory
    std::vector<ImageData> images = load_images_from_directory(image_directory);
    if (images.empty()) {
        std::cerr << "No images found in the directory: " << image_directory << std::endl;
        return 1;
    }

    // Use PCA on the images
    preprocess_images_with_pca(images, pca_ncomp);

    // Convert all images into JSON format save to a file
    json json_data;
    for (const auto& image_data : images) {
        json_data.push_back(image_data_to_json(image_data));
    }
    std::ofstream output_file_stream(output_file);
    if (output_file_stream.is_open()) {
        output_file_stream << json_data.dump(4);
        output_file_stream.close();
        // std::cout << "Image data saved to " << output_file << std::endl;
    } else {
        // std::cerr << "Failed to open output file for writing: " << output_file << std::endl;
        return 1;
    }

    // End the timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total runtime: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
