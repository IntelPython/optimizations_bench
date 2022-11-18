# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

BENCHMARKS = rng
SOURCES = $(addsuffix .c,$(BENCHMARKS))
CC = icx
CLANG_FORMAT = clang-format
CFLAGS += -m64 -fPIC -fomit-frame-pointer -xSSE4.2 -axCORE-AVX2,CORE-AVX512 \
	  -O3 -fp-model fast=2 -fimf-precision=high -prec-sqrt \
	  -fprotect-parens
LDFLAGS += -lmkl_rt

run: $(BENCHMARKS)
	./$<

$(BENCHMARKS): $(SOURCES)
	$(CC) $< $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f $(BENCHMARKS)

format:
	$(CLANG_FORMAT) -i $(SOURCES)

.PHONY: clean run format


