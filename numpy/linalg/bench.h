/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <complex>

using namespace std;

#include "assert.h"
#include "stdlib.h"

#if defined(__INTEL_COMPILER)

#include "mkl.h"

class Random {
  private:
    enum { SEED = 77777 };
    static double const d_zero = 0.0, d_one = 1.0;
    VSLStreamStatePtr stream;

  public:
    Random() {
        int err = vslNewStream(&stream, VSL_BRNG_MT19937, SEED);
        assert(err == VSL_STATUS_OK);
    }
    ~Random() {
        int err = vslDeleteStream(&stream);
        assert(err == VSL_STATUS_OK);
    }
    void init_mat(double *mat, int size) {
        int err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, size, mat,
                                d_zero, d_one);
        assert(err == VSL_STATUS_OK);
    }
};

#else

#include "cblas.h"
#include "lapacke.h"

class Random {
  public:
    void init_mat(double *mat, int size) {
    }
};

static void *mkl_malloc(int size, int align) {
    return malloc(size);
}

static void mkl_free(void *p) {
    free(p);
}
#endif

class Bench {
  private:
    Random random;

  public:
    double *make_random_mat(int size) {
        double *mat = make_mat(size);
        random.init_mat(mat, size);
        return mat;
    }

    double *make_mat(int mat_size) {
        double *mat = (double *) mkl_malloc(mat_size * sizeof(double), 64);
        assert(mat);
        return mat;
    }

    void print_scalar(double x) {
        printf("% .3f", x);
    }

    void print_scalar(complex<double> x) {
        print_scalar(x.real());
        printf(" + ");
        print_scalar(x.imag());
        printf("i");
    }

    template<typename T>
    void print_mat(char mode, T *x, int m, int n) {
        // If mode == 'r', treat it as row-major
        // If mode == 'c', treat it as col-major
        printf("[");
        for (int i = 0; i < m; i++) {
            if (i > 0)
                printf(" ");
            printf("[");
            for (int j = 0; j < n; j++) {
                T num;
                if (mode == 'r')
                    num = x[i*n+j];
                else if (mode == 'c')
                    num = x[j*m+i];
                print_scalar(num);
                if (j < n-1)
                    printf(", ");
            }
            printf("]");
            if (i < m-1)
                printf(",\n");
        }
        printf("]\n");
    }

    template<typename T>
    bool mat_equal(const T *a, const T *b, int n, double tol) {
        for (int i = 0; i < n; i++)
            if (abs(a[i] - b[i]) > tol)
                return false;

        return true;
    }

    template<typename T>
    bool mat_equal(const T *a, const T *b, int n) {
        return mat_equal(a, b, n, 0.00000000000001);
    }

    virtual void make_args(int size) = 0;
    virtual void copy_args() = 0;
    virtual void clean_args() = 0;
    virtual void print_args() = 0;
    virtual void print_result() = 0;
    virtual void compute() = 0;
    virtual bool test(bool verbose) {return false;};
};
