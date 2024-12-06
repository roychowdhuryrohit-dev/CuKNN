#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

void loadJSON(const std::string& filepath, float*& pcaVec, int*& labelVec, int& pcaCount, int& labelCount) {
    nlohmann::json jsonData;

    std::ifstream file(filepath); 

    if (!file.is_open()) {
	std::cerr << "Unable to open the file: " << filepath << std::endl;
	return;
    }

    try {
	file >> jsonData;
	
	labelCount = jsonData.size();
	pcaCount = jsonData[0]["image"].size();

	pcaVec = (float*) malloc(labelCount*pcaCount*sizeof(float));
	labelVec = (int*) malloc(labelCount*sizeof(float));
	
	size_t k = 0;
	for (size_t i = 0; i<labelCount; i++) {
	    
	    labelVec[i] = std::stoi(jsonData[i]["label"].get<std::string>());
	    for (size_t j=0; j<pcaCount; j++) {

		pcaVec[k++] = jsonData[i]["image"][j].get<float>();
	    }
	}


    }  catch (const nlohmann::json::exception& e) {
	std::cerr << "Error parsing JSON: " << e.what() << std::endl;
	exit(1);
    }
}
