/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "qr.h"
#include <cstring>

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

QR::~QR() {
    if (x_mat) mkl_free(x_mat);
    if (x_mat_init) mkl_free(x_mat_init);
    if (r_mat) mkl_free(r_mat);
    if (tau_vec) mkl_free(tau_vec);
}
