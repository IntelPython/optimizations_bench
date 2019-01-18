/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "eig.h"
#include <cstring>
#include <iostream>

Eig::Eig() {
    a_mat = r_mat = vl_mat = vr_mat = wr_vec = wi_vec = 0;
    vr_mat_complex = 0;
}

void Eig::make_args(int size) {
    n = lda = ldvl = ldvr = size;

    mat_size = n * n;

    // input matrix
    a_mat = make_random_mat(mat_size);
    r_mat = make_mat(mat_size);

    // left and right eigenvectors
    vl_mat = make_mat(mat_size);
    vr_mat = make_mat(mat_size);

    // real and imaginary parts of eigenvalues
    wr_vec = make_mat(n);
    wi_vec = make_mat(n);

    // complex eigenvalues and eigenvectors
    w_vec_complex =
        (double _Complex *) mkl_malloc(n * sizeof(*w_vec_complex), 64);
    vr_mat_complex =
        (double _Complex *) mkl_malloc(mat_size * sizeof(*vr_mat_complex), 64);
}

void Eig::copy_args() {
    memcpy(r_mat, a_mat, mat_size * sizeof(*r_mat));
    memset(w_vec_complex, 0, n * sizeof(*w_vec_complex));
    memset(vr_mat_complex, 0, mat_size * sizeof(*vr_mat_complex));
}

void Eig::compute() {
    // Get right eigenvectors and eigenvalues
    int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', n, r_mat, lda, wr_vec,
                             wi_vec, vl_mat, 1, vr_mat, ldvr);
    assert(info == 0);

    // The dgeev call takes up a majority of the time, so running the rest
    // sequentially isn't a big problem.

    // Are all eigenvalues purely real? If so, we need not do anything.
    only_real = true;
    for (int i = 0; i < n; i++) {
        w_vec_complex[i] = CMPLX(wr_vec[i], wi_vec[i]);
        if (wi_vec[i] != 0.0)
            only_real = false;
    }

    if (!only_real) {
        // scipy.linalg.decomp._make_complex_eigvecs
        // LAPACK outputs complex conjugate pairs of eigenvectors as
        // a vector of real parts and a vector of imaginary parts.
        auto *cvec = vr_mat_complex;
        auto *rvec = vr_mat;
        for (int i = 0; i < n; i++, cvec += n, rvec += n) {

            if (wi_vec[i] != 0.0) {
                // Copy real and imaginary parts
                for (int j = 0; j < n; j++) {
                    cvec[j] = CMPLX(rvec[j], rvec[n+j]);
                    cvec[n+j] = CMPLX(rvec[j], -rvec[n+j]);
                }
                i++;
            } else {
                for (int j = 0; j < n; j++) {
                    cvec[j] = CMPLX(rvec[j], 0);
                }
            }
        }
    }
}

Eig::~Eig() {
    if (a_mat)
        mkl_free(a_mat);
    if (vl_mat)
        mkl_free(vl_mat);
    if (vr_mat)
        mkl_free(vr_mat);
    if (wr_vec)
        mkl_free(wr_vec);
    if (wi_vec)
        mkl_free(wi_vec);
    if (w_vec_complex)
        mkl_free(w_vec_complex);
    if (vr_mat_complex)
        mkl_free(vr_mat_complex);
}
