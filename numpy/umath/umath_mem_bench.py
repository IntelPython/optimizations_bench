# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import platform
import warnings
import itertools
import time
import gc
import numpy.core.umath as ncu
import numpy as np
import argparse
from collections import OrderedDict

try:
    from itimer import itime_rdtsc as clock
    clock_name = "itimer rdtsc"
except:
    try:
        # pip install rdtsc
        from rdtsc import get_cycles as clock
        clock_name = "rdtsc"
    except:
        from timeit import default_timer
        clock = lambda: default_timer()*2.e9
        clock_name = "default_timer, note: pip install rdtsc timer for better precision"

argParser = argparse.ArgumentParser(prog="numpy_tests.py",
                                    description="tool to help automating testing",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argParser.add_argument('-l', '--log',       default=None,      help="log")
argParser.add_argument('-i', '--iteration', default='?',       help="iteration")
argParser.add_argument('-p', '--prefix',    default='?',       help="prefix string")
argParser.add_argument('-s', '--size',      default=None,      help="size of array")
argParser.add_argument('-f', '--func',      default=None,      help="single function to test")
argParser.add_argument('-g', '--goal-time', default=1,         help="goal for measured time in ms")
argParser.add_argument('-r', '--repeats',   default=30,        help="repeat experements and get minimum time")
argParser.add_argument('-o', '--offsets',   default=(0,1,2,4), help="Offset from aligned in elements", nargs='+', type=int)
argParser.add_argument('-v', '--verbose',   default=False,     help="print additonal info", action="store_true")
args = argParser.parse_args()

goalTime = float(args.goal_time)/1000.
array_sizes = [1000, 8000, 32000, 100000, 1000000, 2500000] if args.size is None else [int(args.size)]
func_list = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs', 'floor', 'ceil', 'rint', 'trunc', 'sqrt', 'log10',
             'log', 'exp', 'expm1', 'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh', 'arctanh', 'log1p', 'exp2',
             'log2', 'copyto'] if args.func is None else [str(args.func)]
binary_ops =  OrderedDict([('+', np.add), ('*', np.multiply), ('/',  np.true_divide), ('-', np.subtract)]) if args.func is None else {}
scalararraytypes = range(0, 3)
np_types = [np.float64]
totalBinaryMinutes = (len(binary_ops)*len(scalararraytypes)*len(np_types)*len(array_sizes))*2.8/60.0 #  represents time to figure out internalCount and result
totalUnaryMinutes = (len(func_list)*len(np_types)*len(array_sizes))*2.8/60.0 #  represents time to figure out internalCount and result
if args.log != None:
    print("Binary loops have %s operations, %s array/scalar combinations, %s types, %s array sizes.\n" \
          "If each operation takes 0.5 seconds to execute, binary loops will complete in %s minutes. " \
          % (len(binary_ops), str(len(scalararraytypes)), str(len(np_types)), str(len(array_sizes)), totalBinaryMinutes))
    print("Unary loops have %s operations, %s types, %s array sizes.\n" \
          "If each operation takes 0.5 seconds to execute, binary loops will complete in %s minutes. " \
          % (len(func_list), str(len(np_types)), str(len(array_sizes)), totalUnaryMinutes))
    print("Total time it takes to run this Python script: %s minutes" % (totalBinaryMinutes + totalUnaryMinutes))

overheadMin = 0
def emptyF(x, y):
    pass

# we don't want to spend additional time for an indirection (lambda or **kwargs) or branch inside the benchmarking loop
# thus the following two functions are separately defined
def runBenchUnary(func, z, x, internalCount, timer):
    t0 = timer()
    for j in range(internalCount):
        func(x, z) # z is out
    t1 = timer()
    return t0, t1


def runBenchBinary(func, z, x, y, internalCount, timer):
    t0 = timer()
    for j in range(internalCount):
        func(x, y, out=z)
    t1 = timer()
    return t0, t1


def runBench(func, z, x, y = None, internalCount=1, externalCount=int(args.repeats), timer=clock, overhead=overheadMin):
    if y is None:
       run = lambda: runBenchUnary(func, z, x, internalCount, timer)
    else:
       run = lambda: runBenchBinary(func, z, x, y, internalCount, timer)

    clockList = []
    for i in range(externalCount):
        gc.collect()
        gc.disable()
        t0, t1 = run()
        gc.enable()
        clockList.append(t1 - t0)
    #"extreme verbose:", print('#', clockList, min(clockList), '/', internalCount, '-', overhead)
    return min(clockList)/internalCount - overhead

def getInternalCount(func, z, x, y = None):
    internalCount = 1
    runBench(func, z, x, y, 1, externalCount=1) # initial warmup
    for i in range(1, 1000):
        if i > 1:
            internalCount *= 2
        timing = runBench(func, z, x, y, internalCount, externalCount=1, timer=time.time, overhead=0)*internalCount
        if timing >= goalTime:
            timing2 = runBench(func, z, x, y, internalCount, timer=time.time, overhead=0)*internalCount # warmup and double-check
            if timing2 >= goalTime:
               break
    return internalCount, timing2

def checkResults(CPEs):
    if not args.verbose:
       a = CPEs.ravel()
       if(a[0] > np.min(a)*1.15):
          print("warning: anomaly detected, use --verbose and improve stability, [0]:", a[0], " min:", np.min(a))
       sa = np.sort(a)
       if(sa[0] > sa[1]*1.7):
          print("warning: outlier?", sa)

def clOffset(a):
    return int(a.__array_interface__['data'][0]) % 64

overheadMin = runBench(emptyF, 0, 0, internalCount=100000, overhead=0)
print("Overhead time per loop iteration = ",  overheadMin, " clock = ", clock_name)
print("iteration, prefix, op, type, iterations, size, CPE:aligned, CPE:max")

for np_type in np_types:
  zoffsets = xoffsets = yoffsets = args.offsets
  #zoffsets = (0, 0, 0) #2, 4, 6)
  #xoffsets = (0, 0, 0) #2, 4, 6)
  #yoffsets = (0, 0, 0) #2, 4, 6)
  znoffs = len(zoffsets)
  xnoffs = len(xoffsets)
  ynoffs = len(yoffsets)
  for n in array_sizes:
    z0 = np.asarray(np.random.uniform(2.1, 2.9, size=n+27), dtype=np_type) # if binary function is passed with -f
    x0 = np.asarray(np.random.uniform(0.1, 0.9, size=n+27), dtype=np_type) # 27 is a random non-power-of-two > 16
    y0 = np.asarray(np.random.uniform(1.1, 1.9, size=n+27), dtype=np_type) # > 1 for hyperbolic functions
    if clOffset(z0) != 0 or clOffset(x0) != 0 or clOffset(y0) != 0:
        zoff = int((64-clOffset(z0))%64/z0.itemsize)
        xoff = int((64-clOffset(x0))%64/x0.itemsize)
        yoff = int((64-clOffset(y0))%64/y0.itemsize)
        print("Unaligned array data allocation detected, aligning with offsets:", zoff, xoff, yoff)
        z0 = z0[zoff:n+zoff+16]
        x0 = x0[xoff:n+xoff+16]
        y0 = y0[yoff:n+yoff+16]
    assert(clOffset(z0) == 0)
    assert(clOffset(x0) == 0)
    assert(clOffset(y0) == 0)
    assert(np.all(y0 > 0))
    CPEs = np.zeros([znoffs,xnoffs,ynoffs])
    for op in binary_ops:
      np_func = binary_ops[op]
      for scalararraytype in scalararraytypes:
        internalCount, internalTime = getInternalCount(np_func, z0, x0, y0)
        for zi in reversed(range(znoffs)):
          for xi in reversed(range(xnoffs)):
            for yi in reversed(range(ynoffs)):
                clockSum = 0
                clockOverheadSum = 0
                zoff = zoffsets[zi]
                xoff = xoffsets[xi]
                yoff = yoffsets[yi]
                z = z0[zoff:n+zoff]
                x = x0[xoff:n+xoff]
                y = y0[yoff:n+yoff]
                assert(x.dtype == np_type and y.dtype == np_type and z.dtype == np_type)
                assert(x.size == n and y.size == n and z.size == n)
                # Initialize everything before running the measurements
                if scalararraytype == 0:
                    satype = 'C[%d:]=A[%d:]%sB[%d:]' % (zoff, xoff, op, yoff)
                elif scalararraytype == 1:
                    satype = 'C[%d:]=A[%d:]%sscalar' % (zoff, xoff, op)
                    y = y[0]
                elif scalararraytype == 2:
                    satype = 'C[%d:]=scalar%sA[%d:]' % (zoff, op, yoff)
                    y = x
                    x = y[0]

                CPA_min = runBench(np_func, z, x, y, internalCount=internalCount)
                CPE_min = np.true_divide(CPA_min, n)
                CPEs[zi][xi][yi] = CPE_min
                if args.verbose:
                    print(args.iteration, args.prefix, satype, np_type.__name__, '% 6d'%internalCount, '% 7d'%n, '% 4.2f'%CPE_min, sep=', ', flush=True)
        if scalararraytype == 0:
            satype = '    A%sA' % op
        elif scalararraytype == 1:
            satype = '    A%ss' % op
        elif scalararraytype == 2:
            satype = '    s%sA' % op
        checkResults(CPEs)
        print(args.iteration, args.prefix, satype, np_type.__name__, '% 7d'%internalCount, '% 7d'%n, '% 6.2f'%CPEs[0][0][0], '% 6.2f'%np.max(CPEs), sep=', ', flush=True)

    CPEs = np.zeros([znoffs,xnoffs])
    for func in func_list:
        a0 = y0 if func in ['arccosh'] else x0 # otherwise it results in a complex number
        np_func = getattr(np.core.umath, func, getattr(np, func))
        internalCount, internalTime = getInternalCount(np_func, z0, a0)
        for zi in reversed(range(znoffs)):
          for xi in reversed(range(xnoffs)):
            clockSum = 0
            clockOverheadSum = 0
            zoff = zoffsets[zi]
            xoff = xoffsets[xi]
            x = a0[xoff:n+xoff]
            z = z0[zoff:n+zoff]
            if func in ['copyto']: # source and destination are inversed
                x, z = z, x  # TODO: still affects other functions after copyto()

            CPA_min = runBench(np_func, z, x, internalCount=internalCount)
            CPE_min = np.true_divide(CPA_min, n)
            CPEs[zi][xi] = CPE_min
            if args.verbose:
                satype = 'c[%d:]=% 7s(a[%d:])'%(clOffset(z)/z.itemsize, func, clOffset(x)/x.itemsize)
                print(args.iteration, args.prefix, satype, np_type.__name__, '% 6d'%internalCount, '% 7d'%n, '% 4.2f'%CPE_min, sep=', ', flush=True)
        checkResults(CPEs)
        print(args.iteration, args.prefix, '% 7s'%func, np_type.__name__, '% 7d'%internalCount, '% 7d'%n, '% 6.2f'%CPEs[0][0], '% 6.2f'%np.max(CPEs), sep=', ', flush=True)
