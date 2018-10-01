CXX=g++
CPPFLAGS=-Iinclude/obj-parser/src -Iinclude/ -I/opt/cuda/include
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
LDFLAGS=-Linclude/obj-parser/ -L/opt/cuda/lib64
LDLIBS=-lm -lobjparser -lcudart
CUFLAGS= -lineinfo -std=c++14

ifdef DEBUG
CXXFLAGS += -O0 -g
CUFLAGS += -O0 -g
else
CXXFLAGS += -O3
CUFLAGS += -O3
endif

PPMDIR=output
BIN=rasterizer
CUOBJS=$(addprefix src/, gpu_operations.o device_bitset.o)
OBJS=$(addprefix src/, main.o rasterizer.o input_parser.o) $(CUOBJS)

ifdef BENCH
CPPFLAGS += -DBENCH
endif

all: libs $(BIN)

$(BIN): $(OBJS)
	nvcc $(CPPFLAGS) -lineinfo $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cu
	nvcc $(CPPFLAGS) -dc $< -o $@

libs:
	make -C include/

clean:
	$(RM) $(OBJS) $(BIN) $(PPMDIR)/* $(CUOBJS)

clean-all: clean
	make clean -C include/

.PHONY: all libs clean clean-all
