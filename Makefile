# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

# Target executable
TARGET = image_processor
SRCS = image-proc.cpp

# Build rules
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET)
