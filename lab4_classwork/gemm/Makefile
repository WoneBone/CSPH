EXECUTABLE := gemm

CU_FILES   := 

CU_DEPS    :=

CC_FILES   := 

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall 
CUDA=cuda
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart -lcublas
NVCC=/usr/local/$(CUDA)/bin/nvcc
NVCCFLAGS=-arch=compute_89 -g -std=c++11

OBJS= $(OBJDIR)/main.o $(OBJDIR)/gemm.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
