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
CUOBJS=$(addprefix src/, gpu_operations.o)

ifdef GPU
	OBJS += $(CUOBJS)
	CPPFLAGS += -I/opt/cuda/include -DGPU
	LDFLAGS += -L/opt/cuda/lib64
	LDLIBS += -lcudart
endif

all: libs $(BIN)

$(BIN): $(OBJS)
	$(LINK.cc) -o $@ $^ $(LDLIBS)

%.o: %.cu
	nvcc $(CPPFLAGS) -std=c++14 -c $< -o $@

libs:
	make -C include/

clean:
	$(RM) $(OBJS) $(BIN) $(PPMDIR)/* $(CUOBJS)

clean-all: clean
	make clean -C include/

.PHONY: all libs clean clean-all cuda
