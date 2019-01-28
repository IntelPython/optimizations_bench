/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "inv.h"
#include <iostream>
#include <cstring>

static const double x_mat_test[] = {
     0.470442000675409, -0.291482508170914, -0.44183986349643 ,
    -0.176333746005435,  0.007410393215614, -0.739195206041762,
     0.481736547564898,  0.805743972141035, -0.468344563609981
};

static const double r_mat_test[] = {
     1.257751926455496, -1.046174905778333,  0.46462060722515 ,
    -0.931809131348847, -0.01588524263938 ,  0.904147816602771,
    -0.309375898184227, -1.103418649248418, -0.101770412852239
};
static const int test_size = 3;

Inv::Inv() {
    x_mat = 0;
    r_mat = 0;
    ipiv = 0;
}

void Inv::clean_args() {
    if (r_mat)
        mkl_free(r_mat);
    if (ipiv)
        mkl_free(ipiv);
    if (x_mat)
        mkl_free(x_mat);
}

Inv::~Inv() {
    clean_args();
}

void Inv::make_args(int size) {
    n = size;
    m = size;
    lda = size;
    mat_size = m * n;
    int mn_min = min(m, n);

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

void Inv::copy_args() {
    memcpy(r_mat, x_mat, mat_size * sizeof(*r_mat));
}

void Inv::compute() {
    // compute pivoted lu decomposition
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, r_mat, lda, ipiv);
    assert(info == 0);

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, r_mat, lda, ipiv);
    assert(info == 0);
}

bool Inv::test() {
    clean_args();
    make_args(test_size);
    memcpy(x_mat, x_mat_test, mat_size * sizeof(*x_mat));
    copy_args();
    compute();

    return mat_equal(r_mat, r_mat_test, mat_size);
}

void Inv::print_args() {
    std::cout << "Inverse of " << m << "*" << n << " matrix A." << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('r', x_mat, m, n);
}

void Inv::print_result() {
    std::cout << "A**-1 =" << std::endl;
    print_mat('r', r_mat, m, n);
}
