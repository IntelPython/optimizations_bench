/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "inv.h"
#include <cstring>
#include <iostream>

static const int test_size = 5;

Inv::Inv() {
    x_mat = 0;
    x_mat_init = 0;
    ipiv = 0;
}

void Inv::clean_args() {
    if (ipiv)
        mkl_free(ipiv);
    if (x_mat)
        mkl_free(x_mat);
    if (x_mat_init)
        mkl_free(x_mat_init);
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

    copy_args();
}

void Inv::copy_args() {
    memcpy(x_mat, x_mat_init, mat_size * sizeof(*x_mat));
}

void Inv::compute() {
    int info;

    // compute pivoted LU decomposition
    dgetrf(&n, &n, x_mat, &lda, ipiv, &info);
    assert(info == 0);

    // perform workspace query for dgetri
    int lwork = -1;
    double dlwork;
    dgetri(&n, x_mat, &lda, ipiv, &dlwork, &lwork, &info);
    assert(info == 0);

    lwork = (int) (1.01 * dlwork);
    double *work = make_mat(lwork);
    assert(work);

    // actual call to dgetri.
    dgetri(&n, x_mat, &lda, ipiv, work, &lwork, &info);
    assert(info == 0);

    mkl_free(work);
}

bool Inv::test(bool verbose) {
    clean_args();
    make_args(test_size);
    copy_args();
    compute();

    // X * X**-1 = I
    static const double alpha = 1., beta = 0.;
    static const char no_transpose = 'N';
    double *c = make_mat(mat_size);
    dgemm(&no_transpose, &no_transpose, &n, &n, &n, &alpha, x_mat, &n,
          x_mat_init, &n, &beta, c, &n);

    // verify that we got the identity matrix
    bool identity = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double expectation = (i == j) ? 1 : 0;
            if (!mat_equal(&c[i*n + j], &expectation, 1)) {
                identity = false;
                goto cleanup;
            }
        }
    }

cleanup:
    if (verbose) {
        std::cout << "A * A**-1 = (should be identity matrix)" << std::endl;
        print_mat('c', c, n, n);
    }
    mkl_free(c);
    return identity;
}

void Inv::print_args() {
    std::cout << "Inverse of " << n << "*" << n << " matrix A." << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', x_mat_init, n, n);
}

void Inv::print_result() {
    std::cout << "A**-1 =" << std::endl;
    print_mat('c', x_mat, n, n);
}
