/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include "svd.h"

SVD::SVD() {
    a_mat = r_mat = u_mat = vt_mat = s_vec = 0;
}

void
SVD::make_args(int size) {
    n = lda = size;

    mat_size = n*n;

    /* input matrix */
    a_mat = make_random_mat(mat_size);
    r_mat = make_mat(mat_size);

    /* U, V**T matrices */
    u_mat = make_mat(mat_size);
    vt_mat = make_mat(mat_size);

    /* singular values */
    s_vec = make_mat(n);
}

void SVD::copy_args() {
    memcpy(r_mat, a_mat, mat_size * sizeof(*r_mat));
}

void SVD::compute() {
    /* compute svd decomposition */
    int info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', n, n, r_mat, lda, s_vec,
            u_mat, lda, vt_mat, lda);
    assert(info == 0);
}

SVD::~SVD() {
    if (a_mat) mkl_free(a_mat);
    if (u_mat) mkl_free(u_mat);
    if (vt_mat) mkl_free(vt_mat);
    if (s_vec) mkl_free(s_vec);
}
