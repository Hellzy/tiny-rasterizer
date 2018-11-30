CXX=g++
CPPFLAGS=-Iinclude/obj-parser/src -Iinclude/ -I/opt/cuda/include -Iinclude/obj-parser/mtl-parser/src
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
LDFLAGS=
LDLIBS=-lm -Linclude/obj-parser/ -lobjparser -Linclude/obj-parser/mtl-parser -lmtlparser -L/opt/cuda/lib64 -lcudart
CUFLAGS= -std=c++14

ifdef INFO
	CUFLAGS += -res-usage --nvlink-options=--verbose -lineinfo
endif

ifdef DEBUG
CXXFLAGS += -O0 -g
CUFLAGS += -O0 -g -G
else
CXXFLAGS += -O3
CUFLAGS += -O3
endif

PPMDIR=output
BIN=rasterizer
CUOBJS=$(addprefix src/, gpu_operations.o device_vector.o device_lock.o)
OBJS=$(addprefix src/, main.o rasterizer.o input_parser.o) $(CUOBJS)

ifdef BENCH
CPPFLAGS += -DBENCH
endif

all: libs $(BIN)

$(BIN): $(OBJS)
	nvcc --compiler-options '-fPIC' -dlink -o src/link.o $^
	$(LINK.cc) -Wl,-rpath-link,include/obj-parser/mtl-parser -Wl,-rpath,include/obj-parser:include/obj-parser/mtl-parser -o $@ $^ src/link.o $(LDLIBS)

%.o: %.cu
	nvcc $(CPPFLAGS) $(CUFLAGS) --compiler-options '-fPIC' -dc $< -o $@

libs:
	make -C include/

clean:
	$(RM) $(OBJS) $(BIN) $(PPMDIR)/* $(CUOBJS) src/link.o

clean-all: clean
	make clean -C include/

.PHONY: all libs clean clean-all
