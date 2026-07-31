# Build postgen into src/bin/ (git-ignored)
CXX      ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra

BIN_DIR := bin
TARGET  := $(BIN_DIR)/postgen

all: $(TARGET)

$(TARGET): postgen.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
