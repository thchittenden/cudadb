OBJDIR := obj
SRCDIR := src
INCDIR := inc
EXECUTABLE := cudadb
SRC_EXTS := cpp cu

# C++ compiler
CXX := g++ -m64
CXXFLAGS := -O3 -Wall -I $(INCDIR)
LDFLAGS := -L/usr/local/cuda/lib64/ -lcudart

# NVCC compiler
NVCC := nvcc -arch=sm_20 -m64
NVCCFLAGS := -O3 -maxrregcount 20 --restrict -lineinfo -I $(INCDIR)

# find all source files
SRCS := $(foreach ext, $(SRC_EXTS), $(shell find $(SRCDIR)/ -type f -name '*.$(ext)'))

# convert src directories to obj directories
OBJS := $(foreach src, $(SRCS), $(shell echo $(src) | sed -r 's/$(SRCDIR)\/(.*)\.(.*)/$(OBJDIR)\/\1.o/'))

.PHONY: clean debug

default: $(EXECUTABLE)

clean:
		rm -rf $(OBJDIR) $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
		@test -d $(dir $@) || mkdir -p $(dir $@)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
		@test -d $(dir $@) || mkdir -p $(dir $@)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

debug: 
		echo $(SRCS)
		echo $(OBJS)


