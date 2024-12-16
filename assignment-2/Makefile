NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++17 -O2

CXX = g++
CXX_FLAGS = -Wall -Wextra -std=c++17 -O2

SRC_DIR = src
BIN_DIR = bin
INCLUDE_DIR = include

CUDA_SRC = $(SRC_DIR)/assignment-2.cu
OBJ = $(BIN_DIR)/assignment-2_cuda.o

TARGET = $(BIN_DIR)/assignment-2

all: $(TARGET)

$(BIN_DIR): 
	mkdir -p $(BIN_DIR)
 
$(BIN_DIR)/assignment-2_cuda.o: $(CUDA_SRC) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) $^ -o $@

clean:
	rm -rf $(BIN_DIR)
