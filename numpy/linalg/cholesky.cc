/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "cholesky.h"
#include <cstring>
#include <iostream>

static const int test_size = 5;

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

    static const char uplo = 'U';
    static const char trans = 'T';
    static const double alpha = 1.;
    static const double beta = n;
    dsyrk(&uplo, &trans, &n, &n, &alpha, x_mat, &n, &beta, r_mat, &n);

    // we now have r_mat = x_mat * x_mat' + n * np.eye(n)
    // copy back into x_mat
    memcpy(x_mat, r_mat, mat_size * sizeof(*x_mat));
}

void Cholesky::copy_args() {
    // copy moved to compute()
}

void Cholesky::compute() {
    // perform copy here.
    static const int one = 1;
    dcopy(&mat_size, x_mat, &one, r_mat, &one);

    // compute cholesky decomposition
    int info;
    static const char uplo = 'U';
    dpotrf(&uplo, &n, r_mat, &lda, &info);
    assert(info == 0);

    // we only want an upper triangular matrix
    // in scipy, this is done in *potrf wrapper.
    // https://github.com/scipy/scipy/blob/maintenance/1.3.x/scipy/linalg/flapack_pos_def.pyf.src#L85
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            r_mat[i * n + j] = 0.;
        }
    }
}

bool Cholesky::test(bool verbose) {
    clean_args();
    make_args(test_size);
    copy_args();
    compute();

    // verify that r_mat is upper triangular
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (r_mat[i * n + j] != 0.) {
                if (verbose) {
                    std::cerr << "r_mat is not upper triangular!" << std::endl;
                }
                return false;
            }
        }
    }

    // try to reconstruct x_mat from its Cholesky decomposition
    static const double alpha = 1., beta = 0.;
    static const char no_transpose = 'N';
    static const char transpose = 'T';
    double *c = make_mat(mat_size);
    dgemm(&transpose, &no_transpose, &n, &n, &n, &alpha, r_mat, &n,
          r_mat, &n, &beta, c, &n);

    if (verbose) {
        std::cout << "U* * U = (should be equal to A)" << std::endl;
        print_mat('c', c, n, n);
    }
    bool equal = mat_equal(c, x_mat, mat_size);
    mkl_free(c);
    return equal;
}

void Cholesky::print_args() {
    std::cout << "Cholesky decomposition, A = U* * U, of a "
              << "Hermitian positive-definite matrix A." << std::endl;
    std::cout << "A = " << std::endl;
    print_mat('c', x_mat, n, n);
}

void Cholesky::print_result() {
    std::cout << "U = " << std::endl;
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
