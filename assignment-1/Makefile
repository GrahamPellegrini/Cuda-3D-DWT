# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11 -g  # Added -g for debugging

# Directories
SRC_DIR = ./src
BIN_DIR = ./bin

# Source files
SRCS = $(SRC_DIR)/assignment-1.cpp

# Object files
OBJS = $(BIN_DIR)/assignment-1.o

# Executable file
TARGET = $(BIN_DIR)/assignment-1

# Default target
all: $(TARGET)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to build the executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) -o $@ $^

# Rule to compile source files to object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)
