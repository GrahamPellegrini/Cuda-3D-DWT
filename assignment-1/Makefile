# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11 -g

# Directories
SRC_DIR = ./src
BIN_DIR = ./bin
INC_DIR = ./include

# Source files and corresponding object files
SRCS = $(SRC_DIR)/assignment-1.cpp
OBJS = $(BIN_DIR)/assignment-1.o

# Executable files
TARGETS = $(BIN_DIR)/assignment-1

# Default target
all: $(TARGETS)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to build the executable
$(BIN_DIR)/assignment-1: $(BIN_DIR)/assignment-1.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ -I$(INC_DIR)

# Rule to compile source files to object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC_DIR)

# Clean up
clean:
	rm -f $(OBJS) $(TARGETS)
