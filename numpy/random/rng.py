# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import timeit
import sys

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
SEED=123

def main():
    import itertools
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',  required=False, default="IntelPython",     help="Print with each result")
    parser.add_argument('--rng',  required=False, default="mkl", choices=['mkl', 'numpy'], help='RNG implementation\n'
                                                                 'choices:\n'
                                                                 'mkl: mkl_random.RandomState to be used\n'
                                                                 'numpy: numpy.Generator to be used')

    args = parser.parse_args()

    if args.rng == 'mkl':
        try:
            import mkl_random as rnd
            mkl = True
        except (ImportError, ModuleNotFoundError) as e:
            print(str(e))
            sys.exit(1)
    else:
        import numpy.random as rnd
        mkl = False

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
                rs = rnd.RandomState(SEED, brng=brng_name)
            else:
                rs = rnd.Generator(brng_name(seed=SEED))
            t0 = timeit.default_timer()
            for __ in range(INNER_REPS):
                func(rs, (m*100, 1000))
            t1 = timeit.default_timer()
            times_list.append(t1-t0)
        print(f"{args.text},{m*100*1000},{brng_name if mkl else brng_name().__class__.__name__},{sfn},{min(times_list):.5f}")


if __name__ == '__main__':
    main()
