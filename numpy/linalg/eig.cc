/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "eig.h"
#include <cstring>
#include <iostream>

static const double a_mat_test[] = {
    0.470442000675409,  -0.291482508170914, -0.44183986349643,
    -0.176333746005435, 0.007410393215614,  -0.739195206041762,
    0.481736547564898,  0.805743972141035,  -0.468344563609981};

static const std::complex<double> w_vec_complex_test[] = {
    {0.558344640162537, 0.},
    {-0.274418404940748, 0.876285061400947},
    {-0.274418404940748, -0.876285061400947}};

static const std::complex<double> vr_mat_complex_test[] = {
    {-0.870718618937641, 0.},
    {0.491334396303628, 0.},
    {0.020966583991578, 0.},
    {0.12292498568887, 0.296216896839797},
    {0.107429842088307, 0.640393071981176},
    {-0.689565472096294, 0.},
    {0.12292498568887, -0.296216896839797},
    {0.107429842088307, -0.640393071981176},
    {-0.689565472096294, -0.}};

static const int test_size = 3;

// We set these to zero, because the expectation is complex
// eigenvalues and eigenvectors for this test input
static const double wr_vec_test[] = {0., 0., 0.};
static const double vr_mat_test[] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

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
        (std::complex<double> *) mkl_malloc(n * sizeof(*w_vec_complex), 64);
    vr_mat_complex = (std::complex<double> *) mkl_malloc(
        mat_size * sizeof(*vr_mat_complex), 64);
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
        w_vec_complex[i] = std::complex<double>(wr_vec[i], wi_vec[i]);
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
                    cvec[j] = std::complex<double>(rvec[j], rvec[n + j]);
                    cvec[n + j] = std::complex<double>(rvec[j], -rvec[n + j]);
                }
                i++;
                cvec += n;
                rvec += n;
            } else {
                for (int j = 0; j < n; j++) {
                    cvec[j] = std::complex<double>(rvec[j], 0);
                }
            }
        }
    }
}

bool Eig::test() {
    clean_args();
    make_args(test_size);
    memcpy(a_mat, a_mat_test, mat_size * sizeof(*a_mat));
    copy_args();
    compute();

    if (only_real)
        return mat_equal(wr_vec, wr_vec_test, n) &&
               mat_equal(vr_mat, vr_mat_test, mat_size);
    else
        return mat_equal(w_vec_complex, w_vec_complex_test, n) &&
               mat_equal(vr_mat_complex, vr_mat_complex_test, mat_size);
}

void Eig::print_args() {
    std::cout << "Eigenvalues and eigenvectors of " << n << "*" << n
              << " matrix A." << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('c', a_mat, n, n);
}

void Eig::print_result() {
    std::cout << "Eigenvalues =" << std::endl;
    if (only_real)
        print_mat('c', wr_vec, 1, n);
    else
        print_mat('c', w_vec_complex, 1, n);
    std::cout << "Eigenvectors =" << std::endl;
    if (only_real)
        print_mat('c', vr_mat, n, n);
    else
        print_mat('c', vr_mat_complex, n, n);
}

void Eig::clean_args() {
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

Eig::~Eig() {
    clean_args();
}
