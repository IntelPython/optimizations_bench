/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "inv.h"
#include <iostream>
#include <cstring>

static const double x_mat_test[] = {
     0.470442000675409, -0.176333746005435,  0.481736547564898,
    -0.291482508170914,  0.007410393215614,  0.805743972141035,
    -0.44183986349643 , -0.739195206041762, -0.468344563609981
};

static const double r_mat_test[] = {
     1.257751926455496, -0.931809131348847, -0.309375898184227,
    -1.046174905778332, -0.01588524263938 , -1.103418649248418,
     0.46462060722515 ,  0.904147816602771, -0.101770412852238
};
static const int test_size = 3;

Inv::Inv() {
    x_mat = 0;
    x_mat_init = 0;
    r_mat = 0;
    identity = 0;
    ipiv = 0;
}

void Inv::clean_args() {
    if (r_mat)
        mkl_free(r_mat);
    if (ipiv)
        mkl_free(ipiv);
    if (x_mat)
        mkl_free(x_mat);
    if (x_mat_init)
        mkl_free(x_mat_init);
    if (identity)
        mkl_free(identity);
}

Inv::~Inv() {
    clean_args();
}

void Inv::make_args(int size) {
    n = size;
    lda = size;
    mat_size = n * n;

    // input matrix
    x_mat_init = make_random_mat(mat_size);
    x_mat = make_mat(mat_size);

    // list of pivots
    ipiv = (int *) mkl_malloc(n * sizeof(int), 64);
    assert(ipiv);

    // matrix for result
    r_mat = make_mat(mat_size);

    // identity matrix
    identity = make_mat(mat_size);
    memset(identity, 0, mat_size * sizeof(*identity));
    for (int i = 0; i < mat_size; i += n + 1)
        identity[i] = 1;

    copy_args();
}

void Inv::copy_args() {
    memcpy(x_mat, x_mat_init, mat_size * sizeof(*x_mat));
    memcpy(r_mat, identity, mat_size * sizeof(*r_mat));
}

void Inv::compute() {
    // Solve the equation X * X**-1 = I for X**-1.
    int info;
    dgesv(&n, &n, x_mat, &n, ipiv, r_mat, &n, &info);
    assert(info == 0);
}

bool Inv::test() {
    clean_args();
    make_args(test_size);
    copy_args();
    memcpy(x_mat, x_mat_test, mat_size * sizeof(*x_mat));
    compute();

    return mat_equal(r_mat, r_mat_test, mat_size);
}

void Inv::print_args() {
    std::cout << "Inverse of " << n << "*" << n << " matrix A." << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', x_mat, n, n);
}

void Inv::print_result() {
    std::cout << "A**-1 =" << std::endl;
    print_mat('c', r_mat, n, n);
}
