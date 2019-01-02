# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

CC = icc
CFLAGS = -qopenmp -xCORE-AVX2 -axCOMMON-AVX512 -O3 \
		 -g -lmkl_rt

PYTHON ?= python

ACC ?= la
ifeq ($(ACC), ha)
	CFLAGS += -fimf-precision=high -D_VML_ACCURACY_HA_
	CFLAGS += -fp-model precise
endif
ifeq ($(ACC), la)
	CFLAGS += -fimf-precision=medium -D_VML_ACCURACY_LA_
endif
ifeq ($(ACC), ep)
	CFLAGS += -fimf-precision=low -fimf-domain-exclusion=31 -D_VML_ACCURACY_EP_
endif


all: umath_precise
	./umath_precise

clean:
	rm -f umath_precise umath_bench.c

umath_precise: umath_bench.c
	$(CC) umath_bench.c $(CPPFLAGS) $(CFLAGS) -o umath_precise

umath_bench.c: umath_bench.c.src
	$(PYTHON) -m numpy.distutils.conv_template umath_bench.c.src

.PHONY: all clean
