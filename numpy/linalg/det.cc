/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "det.h"
#include <cstring>
#include <iostream>

static const double x_mat_test[] = {
    0.470442000675409,  -0.291482508170914, -0.44183986349643,
    -0.176333746005435, 0.007410393215614,  -0.739195206041762,
    0.481736547564898,  0.805743972141035,  -0.468344563609981};

static const double result_test = 0.4707855751774963;
static const int test_size = 3;

Det::Det() {
    r_mat = x_mat = 0;
    ipiv = 0;
}

void Det::make_args(int size) {
    n = size;
    m = size;
    mn_min = min(m, n);
    lda = size;
    mat_size = m * n;
    assert(m == n);

    // input matrix
    x_mat = make_random_mat(mat_size);

    // list of pivots
    ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
    assert(ipiv);

    // matrix for result
    r_mat = make_mat(mat_size);

    copy_args();
}

void Det::copy_args() {
    memcpy(r_mat, x_mat, mat_size * sizeof(*r_mat));
}

void Det::compute() {
    // compute pivoted lu decomposition
    int info;
    dgetrf(&n, &n, r_mat, &lda, ipiv, &info);
    assert(info == 0);

    double t = 1.0;
    int i, j;
    for (i = 0, j = 0; i < mn_min; i++, j += lda + 1) {
        t *= (ipiv[i] == i + 1) ? r_mat[j] : -r_mat[j];
    }
    result = t;
}

bool Det::test(bool verbose) {
    clean_args();
    make_args(test_size);
    memcpy(x_mat, x_mat_test, mat_size * sizeof(*x_mat));
    copy_args();
    compute();

    return mat_equal(&result, &result_test, 1);
}

void Det::print_args() {
    std::cout << "Determinant of " << n << "x" << n << " matrix A."
              << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', x_mat, n, n);
}

void Det::print_result() {
    std::cout << "det(A) = " << result << std::endl;
}

void Det::clean_args() {
    if (r_mat)
        mkl_free(r_mat);
    if (ipiv)
        mkl_free(ipiv);
    if (x_mat)
        mkl_free(x_mat);
}

Det::~Det() {
    clean_args();
}
