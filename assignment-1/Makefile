# Compiler
CXX = g++

# Directories
SRC_DIR = ./src
BIN_DIR = ./bin
INC_DIR = ./include

# Source files and corresponding object files
SRCS = $(SRC_DIR)/assignment-1.cpp
OBJS = $(BIN_DIR)/assignment-1.o

# Executable files
TARGET = $(BIN_DIR)/assignment-1

# Default target
all: release

# Build configuration flags
DEBUG_FLAGS = -Wall -Wextra -std=c++11 -g -O0
RELEASE_FLAGS = -Wall -Wextra -std=c++11 -O2 -DNDEBUG

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Debug build
debug: CXXFLAGS = $(DEBUG_FLAGS)
debug: $(TARGET)

# Release build
release: CXXFLAGS = $(RELEASE_FLAGS)
release: $(TARGET)

# Rule to build the executable
$(BIN_DIR)/assignment-1: $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ -I$(INC_DIR)

# Rule to compile source files to object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC_DIR)

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)
