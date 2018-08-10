CXX=g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
LDLIBS=-lm
ifdef DEBUG
CXXFLAGS += -g
endif

BIN=rasterizer
OBJS=$(addprefix src/,main.o object.o scene.o rasterizer.o input_parser.o\
	 							triangle.o types.o utils.o)

all: $(BIN)

$(BIN): $(OBJS)
	$(LINK.cc) -o $@ $^ $(LDLIBS)

clean:
	$(RM) $(OBJS) $(BIN)

.PHONY: all clean
