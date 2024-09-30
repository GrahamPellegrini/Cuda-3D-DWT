# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall

# Targets
TARGET = lab2-q1
OBJECTS = lab2-q1.o

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files into object files
lab2-q1.o: lab2-q1.cpp
	$(CXX) $(CXXFLAGS) -c lab2-q1.cpp -o lab2-q1.o

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)
