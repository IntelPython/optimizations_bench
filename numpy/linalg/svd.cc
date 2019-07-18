/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "svd.h"
#include <cstring>
#include <iostream>

static const double x_mat_test[] = {
    0.470442000675409,  -0.291482508170914, -0.44183986349643,
    -0.176333746005435, 0.007410393215614,  -0.739195206041762,
    0.481736547564898,  0.805743972141035,  -0.468344563609981};

static const double u_mat_test[] = {
    0.42804756004831,  0.513259690621653,  -0.743868117558249,
    0.120044160590577, 0.783501687548245,  0.609683938706898,
    0.895748115177918, -0.350270746126503, 0.273762156923104};

static const double s_vec_test[] = {1.144861136346515, 0.758161081776811,
                                    0.542386469976381};

static const double vt_mat_test[] = {
    0.332298742625904, -0.582047668795011, 0.742157703523676,
    0.417682073085115, -0.614694207903402, -0.669098435653028,
    0.845647226373128, 0.532326537024318,  0.038848764550932};

static const int test_size = 3;

SVD::SVD() {
    a_mat = r_mat = u_mat = vt_mat = s_vec = 0;
}

void SVD::make_args(int size) {
    n = lda = size;

    mat_size = n * n;

    // input matrix
    a_mat = make_random_mat(mat_size);
    r_mat = make_mat(mat_size);

    // U, V**T matrices
    u_mat = make_mat(mat_size);
    vt_mat = make_mat(mat_size);

    // singular values
    s_vec = make_mat(n);
}

void SVD::copy_args() {
    // nothing, see compute()
}

void SVD::compute() {

    // copy_args here because we are considering this an "out of place"
    // operation, i.e. overwrite_a=False in scipy call
    memcpy(r_mat, a_mat, mat_size * sizeof(*r_mat));

    // compute svd decomposition
    int info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', n, n, r_mat, lda, s_vec,
                              u_mat, lda, vt_mat, lda);
    assert(info == 0);
}

bool SVD::test() {
    clean_args();
    make_args(test_size);
    memcpy(a_mat, x_mat_test, mat_size * sizeof(*a_mat));
    copy_args();
    compute();

    return mat_equal(u_mat, u_mat_test, mat_size) &&
           mat_equal(s_vec, s_vec_test, n) &&
           mat_equal(vt_mat, vt_mat_test, mat_size);
}

void SVD::print_args() {
    std::cout << "Singular value decomposition of " << n << "*" << n
              << " matrix A." << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', a_mat, n, n);
}

void SVD::print_result() {
    std::cout << "U = " << std::endl;
    print_mat('c', u_mat, n, n);
    std::cout << "Singular values = " << std::endl;
    print_mat('c', s_vec, 1, n);
    std::cout << "V* = " << std::endl;
    print_mat('c', vt_mat, n, n);
}

void SVD::clean_args() {
    if (a_mat)
        mkl_free(a_mat);
    if (u_mat)
        mkl_free(u_mat);
    if (vt_mat)
        mkl_free(vt_mat);
    if (s_vec)
        mkl_free(s_vec);
}

SVD::~SVD() {
    clean_args();
}
