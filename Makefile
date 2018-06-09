CXX=g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
#ifdef DEBUG
CXXFLAGS += -g

BIN=rasterizer
OBJS=$(addprefix src/,main.o object.o)

all: $(BIN)

$(BIN): $(OBJS)
	$(LINK.cc) -o $@ $^ $(LDLIBS)

clean:
	$(RM) $(OBJS) $(BIN)

.PHONY: all clean
