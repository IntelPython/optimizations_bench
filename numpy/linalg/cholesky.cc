/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "cholesky.h"
#include <cstring>
#include <iostream>

static const double x_mat_test[] = {
    5.551063927745538,  0.034194385271978, -0.276508795460738,
    0.034194385271978,  4.704686460853461, 0.087572555571367,
    -0.276508795460738, 0.087572555571367, 6.07658590927362};

static const double r_mat_test[] = {2.356069593145656,
                                    0.,
                                    0.,
                                    0.014513317166631,
                                    2.16898036516661,
                                    0.,
                                    -0.117360198639788,
                                    0.041160281019929,
                                    2.461933858639425};

static const int test_size = 3;

Cholesky::Cholesky() {
    x_mat = r_mat = 0;
}

void Cholesky::make_args(int size) {
    n = lda = size;

    mat_size = n * n;
    int r_size = mat_size;

    // input matrix
    x_mat = make_random_mat(mat_size);

    // matrix for result
    r_mat = make_mat(r_size);
    memset(r_mat, 0, r_size * sizeof(*r_mat));
    // Set r_mat to identity matrix as in python bench
    for (int i = 0; i < n; i++) {
        r_mat[i * n + i] = 1;
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, x_mat, n,
                x_mat, n, n, r_mat, n);

    // we now have r_mat = x_mat * x_mat' + n * np.eye(n)
    // copy back into x_mat
    memcpy(x_mat, r_mat, mat_size * sizeof(*x_mat));
}

void Cholesky::copy_args() {
    memcpy(r_mat, x_mat, mat_size * sizeof(*x_mat));
}

void Cholesky::compute() {
    // compute cholesky decomposition
    int info;
    const char uplo = 'U';
    dpotrf(&uplo, &n, r_mat, &lda, &info);
    assert(info == 0);

    // we only want an upper triangular matrix
    for (int i = 0; i < n - 1; i++) {
        memset(&r_mat[i * n + i + 1], 0, (n - i - 1) * sizeof(*r_mat));
    }
}

bool Cholesky::test() {
    clean_args();
    make_args(test_size);
    memcpy(x_mat, x_mat_test, mat_size * sizeof(*x_mat));
    copy_args();
    compute();

    return mat_equal(r_mat, r_mat_test, mat_size);
}

void Cholesky::print_args() {
    std::cout << "Cholesky decomposition, A = LL*, of a "
              << "Hermitian positive-definite matrix A." << std::endl;
    std::cout << "A = " << std::endl;
    print_mat('c', x_mat, n, n);
}

void Cholesky::print_result() {
    std::cout << "L = " << std::endl;
    print_mat('c', r_mat, n, n);
}

void Cholesky::clean_args() {
    if (r_mat)
        mkl_free(r_mat);
    if (x_mat)
        mkl_free(x_mat);
}

Cholesky::~Cholesky() {
    clean_args();
}
