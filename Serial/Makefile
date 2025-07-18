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

# Debug build (make debug)
debug: CXXFLAGS = $(DEBUG_FLAGS)
debug: $(TARGET)

# Release build (make / make release)
release: CXXFLAGS = $(RELEASE_FLAGS)
release: $(TARGET)

# Rule to build the executable for assignment-1
$(BIN_DIR)/assignment-1: $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ -I$(INC_DIR)

# Rule to compile source files to object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC_DIR)


# Check that no unwanted libraries are included (make check_headers)
check_headers: 
	@echo "Checking for disallowed headers..."
	@if grep -R "include" $(SRC_DIR) | \
		grep -v -e "boost/" -e "<iostream>" -e "<vector>" -e "<string>" -e "<fstream>" -e "<stdexcept>" -e "<cassert>" -e "<cmath>" \
		-e "<cstdlib>" -e "<chrono>" -e "<typeinfo>" -e "<sstream>" -e "<type_traits>" -e "<memory>" -e "<regex>" \
		-e "<tuple>" -e "<functional>" -e "<algorithm>" -e "<numeric>" -e "<locale>" -e "<map>" -e "<set>" -e "<unordered_map>" \
		-e "<unordered_set>" -e "<iterator>" -e "<list>" -e "<deque>" -e "<stack>" -e "<queue>" -e "<array>" -e "<valarray>" \
		-e "<bitset>" -e "<atomic>" > /dev/null; then \
			echo "No disallowed headers found. All good!"; \
		else \
			echo "Disallowed headers found!" && exit 1; \
		fi
# Note Boost C++ has a collected header but for C++ STL ther is no such header so we must individually check for each header 

# Clean up (make clean)
clean: 
	rm -f $(OBJS) $(TARGET)
