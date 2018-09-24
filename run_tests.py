# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
# warning: this is sanity test for Travis CI. The arguments are really bad for real perf testing, use default arguments instead
os.system('miniconda3/envs/intel3/bin/python numpy/umath/umath_mem_bench.py -v --size 10 --goal-time 0.01 --repeats 1')
