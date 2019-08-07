/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "qr.h"
#include <cstring>
#include <iostream>

static const double x_mat_test[] = {
    0.470442000675409,  -0.291482508170914, -0.44183986349643,
    -0.176333746005435, 0.007410393215614,  -0.739195206041762,
    0.481736547564898,  0.805743972141035,  -0.468344563609981};

static const double q_mat_test[] = {
    -0.708166783705387, -0.247310653062924, -0.374882547416749,
    -0.341008805034221, -0.679169383462016, -0.931467229669989,
    -0.280586627216089, -0.252573245441508, 0.97883501187842};

static const double tau_vec_test[] = {1.664309611097381, 1.070875234925678, 0.};

static const double r_mat_test[] = {-0.708166783705387,
                                    0.,
                                    0.,
                                    -0.341008805034221,
                                    -0.679169383462016,
                                    0.,
                                    -0.280586627216089,
                                    -0.252573245441508,
                                    0.97883501187842};

static const int test_size = 3;

QR::QR() {
    x_mat = x_mat_init = r_mat = tau_vec = 0;
}

void QR::make_args(int size) {
    n = lda = size;

    mat_size = n * n;

    // input matrix
    x_mat_init = make_random_mat(mat_size);
    x_mat = make_mat(mat_size);

    // upper triangular output matrix
    r_mat = make_mat(mat_size);
    memset(r_mat, 0, mat_size * sizeof(*r_mat));

    // tau
    tau_vec = make_mat(n);

    copy_args();
}

void QR::copy_args() {
    memcpy(x_mat, x_mat_init, mat_size * sizeof(*x_mat));
}

void QR::compute() {
    // compute qr decomposition
    int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, x_mat, lda, tau_vec);
    assert(info == 0);

    // numpy computes upper triangular part of A even when mode='raw'
    for (int i = 0; i < n; i++) {
        memcpy(&r_mat[i * n], &x_mat[i * n], (i + 1) * sizeof(*r_mat));
    }
}

bool QR::test(bool verbose) {
    clean_args();
    make_args(test_size);
    memcpy(x_mat_init, x_mat_test, mat_size * sizeof(*x_mat));
    copy_args();
    compute();

    return mat_equal(x_mat, q_mat_test, mat_size) &&
           mat_equal(tau_vec, tau_vec_test, n) &&
           mat_equal(r_mat, r_mat_test, mat_size);
}

void QR::print_args() {
    std::cout << "QR decomposition of " << n << "*" << n << " matrix A."
              << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', x_mat_init, n, n);
}

void QR::print_result() {
    std::cout << "LAPACK Q =" << std::endl;
    print_mat('c', x_mat, n, n);
    std::cout << "LAPACK tau =" << std::endl;
    print_mat('c', tau_vec, 1, n);
    std::cout << "R =" << std::endl;
    print_mat('c', r_mat, n, n);
}

void QR::clean_args() {
    if (x_mat)
        mkl_free(x_mat);
    if (x_mat_init)
        mkl_free(x_mat_init);
    if (r_mat)
        mkl_free(r_mat);
    if (tau_vec)
        mkl_free(tau_vec);
}

QR::~QR() {
    clean_args();
}
