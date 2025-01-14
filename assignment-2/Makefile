# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++17
DEBUG_FLAGS = -g -G -DDEBUG
RELEASE_FLAGS = -O2

CXX = g++
CXX_FLAGS = -Wall -Wextra -std=c++17
DEBUG_CXX_FLAGS = -g -DDEBUG
RELEASE_CXX_FLAGS = -O2

# CUDA paths
CUDA_INSTALL_DIR = /usr/local/cuda
CUDA_INCLUDE_PATH = $(CUDA_INSTALL_DIR)/include
CUDA_LIB_PATH = $(CUDA_INSTALL_DIR)/lib64

# Directories
SRC_DIR = src
BIN_DIR = bin
INCLUDE_DIR = include

# Sources
CUDA_SOURCES = $(INCLUDE_DIR)/kernels.cu $(INCLUDE_DIR)/dwt.cu
CUDA_OBJECTS = $(CUDA_SOURCES:$(INCLUDE_DIR)/%.cu=$(BIN_DIR)/%.o)
CPP_SOURCE = $(SRC_DIR)/main.cpp
CPP_OBJECT = $(CPP_SOURCE:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)

# Targets
TARGET_DEBUG = $(BIN_DIR)/main_debug
TARGET_RELEASE = $(BIN_DIR)/main

# Default build is release
all: release

# Create the bin directory if it doesn't exist
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Build for release
release: NVCC_FLAGS += $(RELEASE_FLAGS)
release: CXX_FLAGS += $(RELEASE_CXX_FLAGS)
release: $(TARGET_RELEASE)

# Build for debug
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: CXX_FLAGS += $(DEBUG_CXX_FLAGS)
debug: $(TARGET_DEBUG)

# Compile CUDA sources to object files
$(BIN_DIR)/%.o: $(INCLUDE_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE_PATH) -I$(INCLUDE_DIR) -c $< -o $@

# Compile C++ source to object file
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXX_FLAGS) -I$(CUDA_INCLUDE_PATH) -I$(INCLUDE_DIR) -c $< -o $@

# Link the release target
$(TARGET_RELEASE): $(CPP_OBJECT) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -L$(CUDA_LIB_PATH) -lcudart -o $@ $^

# Link the debug target
$(TARGET_DEBUG): $(CPP_OBJECT) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -L$(CUDA_LIB_PATH) -lcudart -o $@ $^

# Clean up the build
clean:
	rm -rf $(BIN_DIR)
