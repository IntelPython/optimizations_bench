# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

CXX = icc
CXXFLAGS = -O3 -g -xCORE-AVX2 -axCOMMON-AVX512 -qopenmp
override CXXFLAGS += -std=c++11
LDFLAGS = -lmkl_rt -qopenmp

TARGET = linalg
BENCHES = cholesky det dot inv lu qr svd
SOURCES = $(addsuffix .cc,$(BENCHES)) linalg.cc

$(TARGET): $(SOURCES:.cc=.o)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

