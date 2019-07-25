# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

CC = icc
CFLAGS = -qopenmp -xCORE-AVX2 -axCOMMON-AVX512 -O3 \
	 -g -lmkl_rt -Wall -pedantic

ifneq ($(CONDA_PREFIX),)
		CFLAGS += -I$(CONDA_PREFIX)/include
endif

PYTHON ?= python

ACC ?= ha
ifeq ($(ACC), ha)
	CFLAGS += -fimf-precision=high -D_VML_ACCURACY_HA_
endif
ifeq ($(ACC), la)
	CFLAGS += -fimf-precision=medium -D_VML_ACCURACY_LA_
endif
ifeq ($(ACC), ep)
	CFLAGS += -fimf-precision=low -fimf-domain-exclusion=31 -D_VML_ACCURACY_EP_
endif

TARGET=umath_$(ACC)


all: $(TARGET)
	./$(TARGET)

clean:
	rm -f umath_ha umath_la umath_ep umath_bench.c

compile: $(TARGET)


$(TARGET): umath_bench.c
	$(CC) umath_bench.c $(CPPFLAGS) $(CFLAGS) -o $(TARGET)

umath_bench.c: umath_bench.c.src
	$(PYTHON) -m numpy.distutils.conv_template umath_bench.c.src

.PHONY: all clean compile
