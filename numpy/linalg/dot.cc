/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "dot.h"
#include <cstring>

Dot::Dot() {
    a_mat = b_mat = c_mat = r_mat = 0;
}

Dot::~Dot() {
    if (a_mat) mkl_free(a_mat);
    if (b_mat) mkl_free(b_mat);
    if (c_mat) mkl_free(c_mat);
    if (r_mat) mkl_free(r_mat);
}

void Dot::make_args(int size) {
    m = n = k = size;

    a_mat = make_random_mat(m * k);
    b_mat = make_random_mat(k * n);
    c_mat = make_random_mat(m * n);

    r_mat = make_mat(m * n);

    copy_args();
}

void Dot::copy_args() {
    memcpy(r_mat, c_mat, m * n * sizeof(*r_mat));
}

void Dot::compute() {
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, k, alpha,
                a_mat, k, b_mat, n, beta, r_mat, n);
}
