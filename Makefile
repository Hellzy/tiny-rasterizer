CXX=g++
CPPFLAGS=-Iinclude/obj-parser/src -Iinclude/
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
LDFLAGS=-Linclude/obj-parser/
LDLIBS=-lm -lobjparser

ifdef DEBUG
CXXFLAGS += -O0 -g
else
CXXFLAGS += -O3
endif

PPMDIR=output
BIN=rasterizer
OBJS=$(addprefix src/,main.o rasterizer.o input_parser.o)
CUOBJS=$(addprefix src/, gpu_operations.o device_bitset.o)

	OBJS += $(CUOBJS)
	CPPFLAGS += -I/opt/cuda/include
	LDFLAGS += -L/opt/cuda/lib64
	LDLIBS += -lcudart

ifdef BENCH
CPPFLAGS += -DBENCH
endif

all: libs $(BIN)

$(BIN): $(OBJS)
	nvcc $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cu
	nvcc $(CPPFLAGS) -g -std=c++14 -dc $< -o $@

libs:
	make -C include/

clean:
	$(RM) $(OBJS) $(BIN) $(PPMDIR)/* $(CUOBJS)

clean-all: clean
	make clean -C include/

.PHONY: all libs clean clean-all
