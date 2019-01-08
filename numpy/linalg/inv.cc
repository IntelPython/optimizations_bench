/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include "inv.h"

Inv::Inv() {
    x_mat = 0;
    r_mat = 0;
    ipiv = 0;
}

Inv::~Inv() {
    if (r_mat) mkl_free(r_mat);
    if (ipiv) mkl_free(ipiv);
    if (x_mat) mkl_free(x_mat);
}

void Inv::make_args(int size) {
    n = size;
    m = size;
    lda = size;
    mat_size = m*n;
    int mn_min = min(m, n);

    assert(m == n);

    /* input matrix */
    x_mat = make_random_mat(mat_size);

    /* list of pivots */
    ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
    assert(ipiv);

    /* matrix for result */
    r_mat = make_mat(mat_size);
    copy_args();
}

void Inv::copy_args() {
    memcpy(r_mat, x_mat, mat_size * sizeof(*r_mat));
}

void Inv::compute() {
    /* compute pivoted lu decomposition */
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, r_mat, lda, ipiv);
    assert(info == 0);

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, r_mat, lda, ipiv);
    assert(info == 0);
}
