# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Directories
SRC_DIR = ../lab2
BIN_DIR = $(SRC_DIR)/bin
TEMPLATE_DIR = $(SRC_DIR)/template

# Source files
SRCS = $(SRC_DIR)/lab2-q1-grey.cpp \
		$(SRC_DIR)/lab2-q1-multi.cpp \
	   	$(SRC_DIR)/lab2-q2.1.cpp \
	   	$(SRC_DIR)/lab2-q2.2.cpp \
		$(SRC_DIR)/lab2-q3.cpp \



# Object files
OBJS = $(BIN_DIR)/lab2-q1-grey.o \
	   $(BIN_DIR)/lab2-q1-multi.o \
	   $(BIN_DIR)/lab2-q2.1.o \
	   $(BIN_DIR)/lab2-q2.2.o \
	   $(BIN_DIR)/lab2-q3.o \

# Executable files
TARGETS = $(BIN_DIR)/lab2-q1-grey \
		  $(BIN_DIR)/lab2-q1-multi \
		  $(BIN_DIR)/lab2-q2.1 \
		  $(BIN_DIR)/lab2-q2.2 \
		  $(BIN_DIR)/lab2-q3 \

# Default target
all: $(TARGETS)

# Rule for lab2-q1-grey
$(BIN_DIR)/lab2-q1-grey: $(BIN_DIR)/lab2-q1-grey.o
	$(CXX) -o $@ $^

# Rule for lab2-q1-multi
$(BIN_DIR)/lab2-q1-multi: $(BIN_DIR)/lab2-q1-multi.o
	$(CXX) -o $@ $^
	
# Rule for lab2-q2.1
$(BIN_DIR)/lab2-q2.1: $(BIN_DIR)/lab2-q2.1.o
	$(CXX) -o $@ $^

# Rule for lab2-q2.2
$(BIN_DIR)/lab2-q2.2: $(BIN_DIR)/lab2-q2.2.o
	$(CXX) -o $@ $^

# Rule for lab2-q3
$(BIN_DIR)/lab2-q3: $(BIN_DIR)/lab2-q3.o
	$(CXX) -o $@ $^
	
# Rule to compile source files to object files and place them in bin
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


# Clean up
clean:
	rm -f $(OBJS) $(TARGETS)
