CXX=g++ -m64
CXXFLAGS=-I../common -Iobjs/ -O3 -Wall -fPIC
ISPC=ispc
# note: requires AVX2
# disabling AVX2 FMA since it causes a difference in output compared to reference on Mandelbrot 
#ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64 
#ISPCFLAGS=-O3 --target=avx2-i32x16 --arch=x86-64 
ISPCFLAGS=-O3 --target=avx512skx-x32 --arch=x86-64 

APP_NAME=problem3
OBJDIR=objs
COMMONDIR=../common

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/problem3.o $(OBJDIR)/evalSerial.o $(OBJDIR)/eval_ispc.o $(TASKSYS_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm $(TASKSYS_LIB)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/problem3.o: $(OBJDIR)/eval_ispc.h $(COMMONDIR)/CycleTimer.h

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: %.ispc
		$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

