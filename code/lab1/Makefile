# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Directories
SRC_DIR = ../lab1
BIN_DIR = $(SRC_DIR)/bin
TEMPLATE_DIR = $(SRC_DIR)/template

# Source files
SRCS = $(SRC_DIR)/lab1-atomic.cpp \
       $(SRC_DIR)/lab1-thread_returning.cpp \
       $(TEMPLATE_DIR)/lab1-q1.cpp

# Object files
OBJS = $(BIN_DIR)/lab1-atomic.o \
       $(BIN_DIR)/lab1-thread_returning.o \
       $(BIN_DIR)/lab1-q1.o

# Executable files
TARGETS = $(BIN_DIR)/lab1-atomic \
          $(BIN_DIR)/lab1-thread_returning \
          $(BIN_DIR)/lab1-q1

# Default target
all: $(TARGETS)

# Rule for lab1-atomic
$(BIN_DIR)/lab1-atomic: $(BIN_DIR)/lab1-atomic.o
	$(CXX) -o $@ $^

# Rule for lab1-thread_returning
$(BIN_DIR)/lab1-thread_returning: $(BIN_DIR)/lab1-thread_returning.o
	$(CXX) -o $@ $^

# Rule for lab1-q1
$(BIN_DIR)/lab1-q1: $(BIN_DIR)/lab1-q1.o
	$(CXX) -o $@ $^

# Rule to compile source files to object files and place them in bin
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/lab1-q1.o: $(TEMPLATE_DIR)/lab1-q1.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGETS)
