# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Target executables
TARGETS = lab1-q1 lab1-q2 

# Source files with paths
SRCS = lab1-q1.cpp lab1-q2.cpp

# Object files with paths
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGETS)

# Rules for each target
lab1-q1: lab1-q1.o
	$(CXX) -o $@ $^

lab1-q2: lab1-q2.o
	$(CXX) -o $@ $^

# Rule to compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGETS)
