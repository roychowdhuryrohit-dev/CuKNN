# Define paths (adjust based on your project structure)
SRC_DIR = src
BIN_DIR = bin
LIB_DIR = lib

EIGEN_INCLUDE_PATH = $(LIB_DIR)/eigen
STB_IMAGE_PATH = $(LIB_DIR)/stb
NLOHMANN_JSON_PATH = $(LIB_DIR)/json

# Compiler and flags
CXX = g++
CXXFLAGS_BASE = -std=c++17 -O2 -I$(EIGEN_INCLUDE_PATH) -I$(STB_IMAGE_PATH) -I$(NLOHMANN_JSON_PATH)

# OpenMP-specific flags
CXXFLAGS_OMP = $(CXXFLAGS_BASE) -fopenmp -DOMP
LDFLAGS_OMP = -fopenmp -lstdc++fs

# Serial-specific flags
CXXFLAGS_SERIAL = $(CXXFLAGS_BASE) -DSERIAL
LDFLAGS_SERIAL = -lstdc++fs

# Source files and object files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES_OMP = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/omp_%.o)
OBJ_FILES_SERIAL = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/serial_%.o)

# Targets
TARGET_OMP = $(BIN_DIR)/run_pca_omp
TARGET_SERIAL = $(BIN_DIR)/run_pca_serial

# Default target: build both binaries
all: $(TARGET_OMP) $(TARGET_SERIAL)

# Rule to build the OpenMP binary
$(TARGET_OMP): $(OBJ_FILES_OMP)
	$(CXX) $(OBJ_FILES_OMP) -o $@ $(LDFLAGS_OMP)

# Rule to build the Serial binary
$(TARGET_SERIAL): $(OBJ_FILES_SERIAL)
	$(CXX) $(OBJ_FILES_SERIAL) -o $@ $(LDFLAGS_SERIAL)

# Rule to compile .cpp files to .o files for OpenMP
$(BIN_DIR)/omp_%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

# Rule to compile .cpp files to .o files for Serial
$(BIN_DIR)/serial_%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_SERIAL) -c $< -o $@

# Ensure the bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean up object files and binaries
clean:
	rm -f $(OBJ_FILES_OMP) $(OBJ_FILES_SERIAL) $(TARGET_OMP) $(TARGET_SERIAL)

# Phony targets
.PHONY: all clean
