# Compiler configurations
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++17

# Flags
RELEASE_FLAGS = -O2
DEBUG_FLAGS = -g -G -DDEBUG

# Directories
SRC_DIR = src
BIN_DIR = bin
INCLUDE_DIR = include

# Source and object files
CUDA_SRC = $(SRC_DIR)/assignment-2.cu
OBJ = $(BIN_DIR)/assignment-2.o
TARGET = $(BIN_DIR)/assignment-2

# Default build (release)
all: release

# Release build
release: NVCC_FLAGS += $(RELEASE_FLAGS)
release: $(TARGET)

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Profiling build and execution
profile: release
	ncu --set full --target-processes all ./$(TARGET)

nsys-profile: release
	nsys profile -o $(BIN_DIR)/profile ./$(TARGET) file/input.bin file/single_haar.bin 1 0

# Compile and link
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) $^ -o $@

$(OBJ): $(CUDA_SRC) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Create bin directory if it doesn't exist
$(BIN_DIR): 
	mkdir -p $(BIN_DIR)

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)
