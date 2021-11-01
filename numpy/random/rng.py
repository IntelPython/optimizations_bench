# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
try:
    import mkl_random as rnd
    mkl = True
except (ImportError, ModuleNotFoundError):
    import numpy.random as rnd
    mkl = False
import timeit

def sample_uniform(rs, sz):
    rs.uniform(-1, 1, size=sz)


def sample_normal(rs, sz):
    rs.standard_normal(size=sz)


def sample_gamma(rs, sz):
    rs.gamma(5.2, 1, size=sz)


def sample_beta(rs, sz):
    rs.beta(0.7, 2.5, size=sz)


def sample_randint(rs, sz):
    if hasattr(rs, 'randint'):
        rs.randint(0, 100, size=sz, dtype=np.intc)
    elif hasattr(rs, 'integers'):
        rs.integers(0, 100, size=sz, dtype=np.intc)
    else:
        raise RuntimeError


def sample_poisson(rs, sz):
    rs.poisson(7.6, size=sz)


def sample_hypergeom(rs, sz):
    rs.hypergeometric(214, 97, 83, size=sz)


OUTER_REPS=6
INNER_REPS=512

def main():
    import itertools
    if mkl:
        brngs = ['WH', 'PHILOX4X32X10', 'MT2203', 'MCG59', 'MCG31', 'MT19937', 'MRG32K3A', 'SFMT19937', 'R250']
    else:
        brngs = [np.random.MT19937, np.random.Philox]
    samplers = {'uniform': sample_uniform, 'normal': sample_normal, 'gamma': sample_gamma, 'beta': sample_beta,
                'randint': sample_randint, 'poisson': sample_poisson, 'hypergeom': sample_hypergeom}
    multipliers = {'uniform': 10, 'normal': 2, 'gamma': 1, 'beta': 1, 'randint': 10, 'poisson': 5, 'hypergeom': 1}
    for brng_name, sfn in itertools.product(brngs, samplers.keys()):
        func = samplers[sfn]
        m = multipliers[sfn]
        times_list = []
        for __ in range(OUTER_REPS):
            if mkl:
                rs = rnd.RandomState(123, brng=brng_name)
            else:
                rs = rnd.Generator(brng_name(seed=123))
            t0 = timeit.default_timer()
            for __ in range(INNER_REPS):
                func(rs, (m*100, 1000))
            t1 = timeit.default_timer()
            times_list.append(t1-t0)
        print("IntelPython,{brng_name},{dist_name},{time:.5f}".format(brng_name=brng_name, dist_name=sfn, time=min(times_list)))


if __name__ == '__main__':
    main()
