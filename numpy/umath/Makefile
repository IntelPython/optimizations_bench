# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

CC = icc
CFLAGS = -qopenmp -xCORE-AVX2 -axCOMMON-AVX512 -O3 \
		 -g -fp-model precise -fimf-precision=high -lmkl_rt

PYTHON ?= python

all: umath_precise
	./umath_precise

clean:
	rm -f umath_precise umath_bench.c

umath_precise: umath_bench.c
	$(CC) umath_bench.c $(CPPFLAGS) $(CFLAGS) -o umath_precise

umath_bench.c: umath_bench.c.src
	$(PYTHON) -m numpy.distutils.conv_template umath_bench.c.src

.PHONY: all clean
