/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "lu.h"
#include <cstring>
#include <iostream>

static const double x_mat_test[] = {
    0.470442000675409,  -0.291482508170914, -0.44183986349643,
    -0.176333746005435, 0.007410393215614,  -0.739195206041762,
    0.481736547564898,  0.805743972141035,  -0.468344563609981};

static const double p_mat_test[] = {1., 0., 0., 0., 0., 1., 0., 1., 0.};

static const double l_mat_test[] = {
    1., -0.9392015654684, -0.619592867457488, 0., 1., 0.112559485278225, 0., 0.,
    1.};

static const double u_mat_test[] = {0.470442000675409,
                                    0.,
                                    0.,
                                    -0.176333746005435,
                                    -0.904808136334974,
                                    0.,
                                    0.481736547564898,
                                    -0.015896843993687,
                                    1.106013841583318};

static const int test_size = 3;

LU::LU() {
    x_mat = r_mat = l_mat = u_mat = p_mat = 0;
}

void LU::make_args(int size) {
    m = n = lda = size;

    mat_size = m * n;
    int r_size = mat_size;

    mn_min = min(m, n);
    l_size = m * mn_min;
    u_size = mn_min * n;
    p_size = m * m;

    // input matrix
    x_mat = make_random_mat(mat_size);

    // list of pivots
    ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
    assert(ipiv);

    // matrix for result
    r_mat = make_mat(r_size);

    // lower triangular matrix
    l_mat = make_random_mat(l_size);

    // upper triangular matrix
    u_mat = make_random_mat(u_size);

    // permutation matrix
    p_mat = make_random_mat(p_size);

    copy_args();
    lda = m + n - lda;
}

void LU::copy_args() {
    memcpy(r_mat, x_mat, mat_size * sizeof(*r_mat));
}

void LU::compute() {
    // compute pivoted lu decomposition
    int info;
    dgetrf(&m, &n, r_mat, &lda, ipiv, &info);
    assert(info == 0);

    int ld_l = m;
    int ld_u = mn_min;
    int ld_p = m;
    memset(l_mat, 0, l_size * sizeof(double));
    memset(u_mat, 0, u_size * sizeof(double));

    // extract L and U matrix elements from r_mat
    // https://github.com/scipy/scipy/blob/maintenance/1.3.x/scipy/linalg/src/lu.f#L31-L45
    for (int j = 0; j < mn_min; j++) {
        l_mat[j * ld_l + j] = 1.;
#pragma ivdep
        for (int i = j + 1; i < m; i++) {
            l_mat[j * ld_l + i] = r_mat[j * lda + i];
        }
    }

    for (int j = 0; j < mn_min; j++) {
#pragma ivdep
        for (int i = 0; i <= j; i++) {
            u_mat[j * ld_u + i] = r_mat[j * lda + i];
        }
    }

    for (int j = mn_min; j < n; j++) {
#pragma ivdep
        for (int i = 0; i < mn_min; i++) {
            u_mat[j * ld_u + i] = r_mat[j * lda + i];
        }
    }

    // permute_l=False
    // make a diagonal matrix (m,m)...
    memset(p_mat, 0, p_size * sizeof(double));
    for (int i = 0; i < m; i++)
        p_mat[i * (m + 1)] = 1.0;

    // ...and permute that matrix instead of L
    static const int one = 1, neg_one = -1;
    dlaswp(&m, p_mat, &m, &one, &mn_min, ipiv, &neg_one);
    assert(info == 0);
}

bool LU::test(bool verbose) {
    clean_args();
    make_args(test_size);
    memcpy(x_mat, x_mat_test, mat_size * sizeof(*x_mat));
    copy_args();
    compute();

    return mat_equal(p_mat, p_mat_test, mat_size) &&
           mat_equal(l_mat, l_mat_test, mat_size) &&
           mat_equal(u_mat, u_mat_test, mat_size);
}

void LU::print_args() {
    std::cout << "LU decomposition P*L*U of " << m << "*" << n << " matrix A."
              << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', x_mat, m, n);
}

void LU::print_result() {
    std::cout << "P =" << std::endl;
    print_mat('c', p_mat, m, m);
    std::cout << "L =" << std::endl;
    print_mat('c', l_mat, m, mn_min);
    std::cout << "U =" << std::endl;
    print_mat('c', u_mat, mn_min, n);
}

void LU::clean_args() {
    if (l_mat)
        mkl_free(l_mat);
    if (u_mat)
        mkl_free(u_mat);
    if (r_mat)
        mkl_free(r_mat);
    if (p_mat)
        mkl_free(p_mat);

    if (ipiv)
        mkl_free(ipiv);
    if (x_mat)
        mkl_free(x_mat);
}

LU::~LU() {
    clean_args();
}
