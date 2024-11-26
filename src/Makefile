# Define paths (will be to be changed depending on library locations)
EIGEN_INCLUDE_PATH = /WAVE/users2/unix/klor/ycho_lab/kimsong_lor/clang10_venv/include/eigen3
STB_IMAGE_PATH = /WAVE/users2/unix/klor/ycho_lab/kimsong_lor/csen_319/stb
NLOHMANN_JSON_PATH = /WAVE/users2/unix/klor/ycho_lab/kimsong_lor/clang10_venv/lib/python3.9/site-packages/nlohmann_json/include

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp -I$(EIGEN_INCLUDE_PATH) -I$(STB_IMAGE_PATH) -I$(NLOHMANN_JSON_PATH)

# Linker flags
LDFLAGS = -fopenmp

# Source files
SRC_FILES = main.cpp dataset_preprocessing.cpp

# Object files
OBJ_FILES = $(SRC_FILES:.cpp=.o)

# Output executable
TARGET = run_pca

# Default target to build the project
all: $(TARGET)

# Rule to link object files and create the final executable
$(TARGET): $(OBJ_FILES)
	$(CXX) $(OBJ_FILES) -o $(TARGET) $(LDFLAGS)

# Rule to compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -f $(OBJ_FILES) $(TARGET)

# Phony targets
.PHONY: all clean
