/*
 * Copyright (C) 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "mkl.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define SAMPLE_SIZE 100000
#define INNER_REPS 512
#define OUTER_REPS 6

/* mkl_random.uniform(-1,1) */
extern void sample_uniform(VSLStreamStatePtr stream, MKL_INT sample_size) {
    int err;
    double *x;
    double a = -1.0, b = 1.0;
    x = (double *) mkl_malloc(sizeof(double)*sample_size, 64);

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, sample_size, x, a, b);
    if (err != VSL_STATUS_OK) {
	printf("Uniform RNG error code: %d\n", err);
    }

    mkl_free(x);
}


/* mkl_random.standard_normal */
extern void sample_normal(VSLStreamStatePtr stream, MKL_INT sample_size) {
    int err;
    double *x;
    double mu_zero = 0.0, sigma_one = 1.0;

    x = (double *) mkl_malloc(sizeof(double)*sample_size, 64);
    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, sample_size, x,
			mu_zero, sigma_one);
    if (err != VSL_STATUS_OK) {
	printf("Normal RNG error code: %d\n", err);
    }

    mkl_free(x);
}


/* mkl_random.gamma(5.2, 1) */
extern void sample_gamma(VSLStreamStatePtr stream, MKL_INT sample_size) {
    int err;
    double *x;
    double shape_par = 5.2, scale_one = 1.0, loc_zero = 0.0;

    x = (double *) mkl_malloc(sizeof(double)*sample_size, 64);
    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, stream, sample_size, x,
		     shape_par, loc_zero, scale_one);
    if (err != VSL_STATUS_OK) {
	printf("Gamma RNG error code: %d\n", err);
    }

    mkl_free(x);
}


/* mkl_random.beta(0.7, 2.5) */
extern void sample_beta(VSLStreamStatePtr stream, MKL_INT sample_size) {
    int err;
    double *x;
    double shape_par1 = 0.7, shape_par2 = 2.5;
    double loc_zero = 0.0, scale_one = 1.0;

    x = (double *) mkl_malloc(sizeof(double)*sample_size, 64);
    err = vdRngBeta(VSL_RNG_METHOD_BETA_CJA_ACCURATE, stream, sample_size, x,
		    shape_par1, shape_par2, loc_zero, scale_one);
    if (err != VSL_STATUS_OK) {
	printf("Beta RNG error code: %d\n", err);
    }

    mkl_free(x);
}


/* mkl_random.randint(0,100) */
extern void sample_randint(VSLStreamStatePtr stream, MKL_INT sample_size) {
    MKL_INT *x;
    int err;
    MKL_INT a = 0, b = 100;

    x = (MKL_INT *) mkl_malloc(sizeof(MKL_INT) * sample_size, 64);
    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, sample_size, x, a, b);
    if (err != VSL_STATUS_OK) {
	printf("RandInt RNG error code: %d\n", err);
    }
    mkl_free(x);
}

/* mkl_random.poisson(7.2) */
extern void sample_poisson(VSLStreamStatePtr stream, MKL_INT sample_size) {
    MKL_INT *x;
    int err;
    double rate = 7.2;

    x = (MKL_INT *) mkl_malloc(sizeof(MKL_INT) * sample_size, 64);
    err = viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, stream, sample_size, x, rate);
    if (err != VSL_STATUS_OK) {
	printf("Poisson RNG error code: %d\n", err);
    }
    mkl_free(x);
}

/* mkl_random.hypergeometric(n_good=214, n_bad=97, n_sample=83) */
extern void sample_hypergeom(VSLStreamStatePtr stream, MKL_INT sample_size) {
    MKL_INT *x;
    int err;
    MKL_INT el=214 + 97, es = 83, em = 214;

    x = (MKL_INT *) mkl_malloc(sizeof(MKL_INT) * sample_size, 64);
    err = viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, stream, sample_size, x, el, es, em);
    if (err != VSL_STATUS_OK) {
	printf("RandInt RNG error code: %d\n", err);
    }

    mkl_free(x);
}

#define BRNGS_LEN 9
const MKL_INT brngs[BRNGS_LEN] = {
    VSL_BRNG_WH,
    VSL_BRNG_MT19937,
    VSL_BRNG_SFMT19937,
    VSL_BRNG_MT2203,
    VSL_BRNG_R250,
    VSL_BRNG_MCG31,
    VSL_BRNG_MCG59,
    VSL_BRNG_MRG32K3A,
    VSL_BRNG_PHILOX4X32X10
};
const char* brng_names[BRNGS_LEN] = {
    "WH", "MT19937", "SFMT19937", "MT2203", "R250", "MCG31", "MCG59", "MRG32K3A", "PHILOX4X32X10"
};

typedef void (*DistributionSampler)(VSLStreamStatePtr stream, MKL_INT sample_size);

#define FN_LEN 7
const DistributionSampler fns[FN_LEN] = {
    &sample_uniform,
    &sample_normal,
    &sample_gamma,
    &sample_beta,
    &sample_randint,
    &sample_poisson,
    &sample_hypergeom
};

const char* dist_names[FN_LEN] = {
    "uniform", "normal", "gamma", "beta", "randint", "poisson", "hypergeom"
};

const MKL_INT dist_sample_sizes[FN_LEN] = {
    10 * SAMPLE_SIZE, /* uniform */
    2 * SAMPLE_SIZE,
    SAMPLE_SIZE,
    SAMPLE_SIZE,
    10 * SAMPLE_SIZE, /* randint */
    5 * SAMPLE_SIZE,  /* poisson */
    SAMPLE_SIZE
};

int main(void) {
    VSLStreamStatePtr stream;
    int err, outer_it, inner_it, brng_idx, fn_idx;
    double times[OUTER_REPS];

    for (brng_idx=0; brng_idx < BRNGS_LEN; brng_idx++) {
	for (fn_idx = 0; fn_idx < FN_LEN; fn_idx++) {
	    DistributionSampler sampling_fn = fns[fn_idx];
	    MKL_INT sz = dist_sample_sizes[fn_idx];

	    for(outer_it=0; outer_it < OUTER_REPS; outer_it++) {
		struct timespec ts_start, ts_finish;

		err = vslNewStream(&stream, brngs[brng_idx], 123);
		if (err != VSL_STATUS_OK) {
		    printf("PANIC: abandon ship... \n");
		}

		clock_gettime(CLOCK_MONOTONIC, &ts_start);
		for(inner_it=0; inner_it < INNER_REPS; inner_it++) {
		    (*sampling_fn)(stream, sz);
		}
		clock_gettime(CLOCK_MONOTONIC, &ts_finish);

		times[outer_it] = (ts_finish.tv_sec - ts_start.tv_sec) + (ts_finish.tv_nsec - ts_start.tv_nsec) * (1e-9);
		vslDeleteStream(&stream);
	    }


	    {
		double min_time = times[0];
		int i;
		for (i=1; i < OUTER_REPS; i++)
		    if (times[i] < min_time) min_time = times[i];

		printf("Native-C,%s,%s,%.5f\n",brng_names[brng_idx], dist_names[fn_idx], min_time);
	    }
	}
    }

    return 0;
}
