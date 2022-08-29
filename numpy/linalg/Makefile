# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

CXX = icx
CXXFLAGS = -O3 -g -xCORE-AVX2 -xCOMMON-AVX512 \
	   -qopt-report=5 -qopt-report-phase=openmp,par,vec
LDFLAGS = -lmkl_rt

TARGET = linalg
BENCHES = cholesky det dot eig inv lu qr svd
SOURCES = $(addsuffix .cc,$(BENCHES)) linalg.cc

ifneq ($(CONDA_PREFIX),)
	LDFLAGS += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)
	CXXFLAGS += -I$(CONDA_PREFIX)/include
endif

override CXXFLAGS += -std=c++11

CLANG_FORMAT = clang-format

$(TARGET): $(SOURCES:.cc=.o)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.optrpt

format:
	$(CLANG_FORMAT) -i $(SOURCES)

.PHONY: clean format

