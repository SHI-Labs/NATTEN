# Copied with modifications from 
# https://github.com/ColfaxResearch/cfx-article-src/blob/master/tma/Makefile

CUTLASS_DIR=../../../../../../third_party/cutlass
HOPPER_DIR=..
THIS_DIR=.
CXX=nvcc
APP=main

CXXFLAGS=--generate-code=arch=compute_80,code=sm_80 -std=c++17 -O3 --ftemplate-backtrace-limit=0 \
				 --use_fast_math --resource-usage -lineinfo -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG --expt-relaxed-constexpr
INCLUDES=-I${CUTLASS_DIR}/include  -I${CUTLASS_DIR}/tools/util/include -I${HOPPER_DIR} -I${THIS_DIR}

LDFLAGS=

LDLIBS=-lcuda

OBJECTS = main.o

.SUFFIXES: .o .cu

default: clean $(APP)

$(APP): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

main.o:
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o "$@" main.cu

clean:
	rm -f $(OBJECTS) $(APP)
