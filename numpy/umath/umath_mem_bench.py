# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import sys
import platform
import warnings
import itertools
import time
import gc
import numpy as np
import numpy.core.umath
import argparse

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
        clock_name = "default_timer, note: 'pip install rdtsc' for better precision"

argParser = argparse.ArgumentParser(prog="numpy_tests.py",
                                    description="tool to help automating testing",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def_sizes = [1000, 8000, 32000, 100000, 1000000, 2500000]
def_funcs = ['+', '-', '*', '/', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    #'fabs', 'floor', 'ceil', 'rint', 'trunc',
    'sqrt', 'log10', 'log', 'exp', 'expm1', 'arcsin', 'erf',
    'arccos', 'arctan', 'arcsinh', 'arccosh', 'arctanh', 'log1p', 'exp2', 'log2', 'copyto']
def_impls = ['numpy', 'numexpr', 'numba']

argParser.add_argument('-l', '--log',       default=None,      help="log")
argParser.add_argument('-p', '--prefix',    default='@',       help="prefix string")
argParser.add_argument('-s', '--size',      default=def_sizes, help="size of array", nargs='+', type=int)
argParser.add_argument('-f', '--func',      default=def_funcs, help="function(s) to test", nargs='+', type=str)
argParser.add_argument('-m', '--impl',      default=def_impls, help="implementation(s) to test", nargs='+', type=str)
argParser.add_argument('-g', '--goal-time', default=1,         help="goal for measured time in ms")
argParser.add_argument('-r', '--repeats',   default=30,        help="repeat experements and get minimum time")
argParser.add_argument('-o', '--offsets',   default=(0,1,2,4), help="Offset from aligned in elements", nargs='+', type=int)
argParser.add_argument(    '--fast-math',   default=False,     help="Enable fast math mode", action="store_true")
argParser.add_argument('-v', '--verbose',   default=False,     help="print additonal info", action="store_true")
args = argParser.parse_args()
assert type(args.func) is list
assert type(args.impl) is list
assert type(args.size) is list

for impl in args.impl:
    if impl == "numexpr":
        import numexpr
    if impl == "numba":
        import numba, math
        if "OMP_NUM_THREADS" in os.environ:
            run_par = numba.config.NUMBA_DEFAULT_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
        run_par = numba.config.NUMBA_DEFAULT_NUM_THREADS > 1

goalTime = float(args.goal_time)/1000.
scalararraytypes = range(0, 3)
np_types = [np.float64]
overheadMin = 0

def numbaKernel(params, code, sigs):
    n = "numba_%X"%abs(hash(code))
    r = "numba.prange" if run_par else "range"
    exec("""def {n}({params},out):
              for i in {r}(out.size):
                out[i] = {code}
         """.format(**locals()))
    return numba.njit(sigs, parallel=run_par, fastmath=args.fast_math)(locals()[n])

def getBinaryFuncImpl(func, impl, scalar):
    if impl == "numexpr":
        return lambda x,y,out: numexpr.evaluate("x %s y"%func, out=out)
    elif impl == 'numpy':
        return {'+': np.add, '*': np.multiply, '/':  np.true_divide, '-': np.subtract}[func]
    elif impl == 'numba':
      if scalar == 0:
        return numbaKernel("x,y", "x[i] %s y[i]"%func, "(f8[::1],f8[::1],f8[::1])")
      elif scalar == 1:
        return numbaKernel("x,y", "x[i] %s y"%func, "(f8[::1],f8,f8[::1])")
      elif scalar == 2:
        return numbaKernel("x,y", "x %s y[i]"%func, "(f8,f8[::1],f8[::1])")
    raise NotImplementedError(impl)

def getUnaryFuncImpl(func, impl):
    if impl == "numexpr":
        if func == 'invsqrt':
            return lambda x,out: numexpr.evaluate("1/sqrt(x)", out=out)
        else:
            return lambda x,out: numexpr.evaluate("%s(x)"%func, out=out)
    elif impl == 'numba':
        if func == 'invsqrt':
            return numbaKernel("x", "1./math.sqrt(x[i])", "(f8[::1],f8[::1])")
        else:
            return numbaKernel("x", "math.%s(x[i])"%func, "(f8[::1],f8[::1])")
    elif impl == 'numpy':
        try:
            ufunc = getattr(np.core.umath, func, None)
            if not ufunc:
                ufunc = getattr(np, func)
            return ufunc
        except:
            if func == "erf":
                from scipy.special import erf
                return erf
            elif func == "invsqrt":
                return lambda x, y: 1/np.sqrt(x, y)
            else:
                raise
    raise NotImplementedError(impl)


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
print("Prefix,Implementation,Function,Type,Iterations,Size,CPE:aligned,CPE:max")

for np_type in np_types:
  zoffsets = xoffsets = yoffsets = args.offsets
  znoffs = len(zoffsets)
  xnoffs = len(xoffsets)
  ynoffs = len(yoffsets)
  for n in args.size:
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

    for impl in args.impl:
      for func in args.func:
        try:
            if len(func) <= 2: # binary operation
              op = func
              CPEs = np.zeros([znoffs,xnoffs,ynoffs])
              for scalararraytype in scalararraytypes:
                np_func = getBinaryFuncImpl(func, impl, scalararraytype)
                if scalararraytype == 0:
                    x,y,z = x0,y0,z0
                elif scalararraytype == 1:
                    x,y,z = x0,y0[0],z0
                elif scalararraytype == 2:
                    x,y,z = x0[0],y0,z0
                internalCount, internalTime = getInternalCount(np_func, z, x, y)
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
                            #y = x  y=y[0]
                            x = x[0]

                        CPA_min = runBench(np_func, z, x, y, internalCount=internalCount)
                        CPE_min = np.true_divide(CPA_min, n)
                        CPEs[zi][xi][yi] = CPE_min
                        if args.verbose:
                            print(args.prefix, '% 7s'%impl, satype, np_type.__name__, '% 6d'%internalCount, '% 7d'%n, '% 4.2f'%CPE_min, sep=', ', flush=True)
                if scalararraytype == 0:
                    satype = ' array%sarray' % op
                elif scalararraytype == 1:
                    satype = 'array%sscalar' % op
                elif scalararraytype == 2:
                    satype = 'scalar%sarray' % op
                checkResults(CPEs)
                print(args.prefix, '% 7s'%impl, '% 12s'%satype, np_type.__name__, '% 7d'%internalCount, '% 7d'%n, '% 6.2f'%CPEs[0][0][0], '% 6.2f'%np.max(CPEs), sep=', ', flush=True)
            else: # unary operation
                np_func = getUnaryFuncImpl(func, impl)
                CPEs = np.zeros([znoffs,xnoffs])
                a0 = y0 if func in ['arccosh'] else x0 # otherwise it results in a complex number
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
                        print(args.prefix, '% 7s'%impl, satype, np_type.__name__, '% 6d'%internalCount, '% 7d'%n, '% 4.2f'%CPE_min, sep=', ', flush=True)
                checkResults(CPEs)
                print(args.prefix, '% 7s'%impl, '% 12s'%func, np_type.__name__, '% 7d'%internalCount, '% 7d'%n, '% 6.2f'%CPEs[0][0], '% 6.2f'%np.max(CPEs), sep=', ', flush=True)
        except Exception as e:
            print("Failed while executing %s for %s: %s" % (func, impl, str(e)))
