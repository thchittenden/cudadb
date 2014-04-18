OBJDIR := obj
SRCDIR := src
INCDIR := inc
TSTDIR := tst
EXECUTABLE := cudadb
SRC_EXTS := cpp cu

ASSERTIONS = false
DEBUG = true

# C++ compiler
CXX := g++
CXXFLAGS := -O3 -Wall -I $(INCDIR)
LDFLAGS := -L/usr/local/cuda/lib64/ -lcudart

# NVCC compiler
NVCC := nvcc -arch=sm_20
NVCCFLAGS := -O3 --restrict -lineinfo -I $(INCDIR)

# add debug flag if necessary
ifeq ($(DEBUG),true)
	NVCCFLAGS := $(NVCCFLAGS) -D__DEBUG__ -g -G
	CXXFLAGS := $(CXXFLAGS) -D__DEBUG__ -gstabs+
endif

ifeq ($(ASSERTIONS),true)
	NVCCFLAGS := $(NVCCFLAGS) -D__ALWAYS_DEBUG__
endif

# find all source files
SRCS := $(foreach ext, $(SRC_EXTS), $(shell find $(SRCDIR)/ -type f -name '*.$(ext)'))

# convert src directories to obj directories
OBJS := $(foreach src, $(SRCS), $(shell echo $(src) | sed -r 's/$(SRCDIR)\/(.*)\.(.*)/$(OBJDIR)\/\1.o/'))

# get 
TSTS := $(shell find $(TSTDIR)/ -type f -name '*.cpp')
TSTS_EXEC := $(basename $(TSTS))

.PHONY: clean tests debug

default: $(EXECUTABLE)

clean:
		rm -rf $(OBJDIR) $(EXECUTABLE) $(TSTS_EXEC)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
		@test -d $(dir $@) || mkdir -p $(dir $@)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
		@test -d $(dir $@) || mkdir -p $(dir $@)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(TSTDIR)/%: $(TSTDIR)/%.cpp $(OBJS)
		$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS) $<

tests: $(TSTS_EXEC)

debug: 
		echo $(SRCS)
		echo $(OBJS)


