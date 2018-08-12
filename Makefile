CXX=g++
CPPFLAGS=-Iinclude/obj-parser/src -Iinclude/
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
LDFLAGS=-Linclude/obj-parser/
LDLIBS=-lm -lobjparser
ifdef DEBUG
CXXFLAGS += -g
endif

PPMDIR=output
BIN=rasterizer
OBJS=$(addprefix src/,main.o rasterizer.o input_parser.o)

all: libs $(BIN)

$(BIN): $(OBJS)
	$(LINK.cc) -o $@ $^ $(LDLIBS)

libs:
	make -C include/

clean:
	$(RM) $(OBJS) $(BIN) $(PPMDIR)/*

clean-all: clean
	make clean -C include/

.PHONY: all libs clean clean-all
