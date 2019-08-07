/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "dot.h"
#include <cstring>
#include <iostream>

static const double a_mat_test[] = {
    0.470442000675409,  -0.176333746005435, 0.481736547564898,
    -0.291482508170914, 0.007410393215614,  0.805743972141035,
    -0.44183986349643,  -0.739195206041762, -0.468344563609981};

static const double b_mat_test[] = {
    -0.279551412836935, -1.866235595807669, 0.949267732307811,
    0.393910888693485,  0.357832357041521,  1.430099195743549,
    -0.202579028296422, -1.225349132812327, 0.535350173863021};

static const double r_mat_test[] = {
    -0.298562230242867, -1.531348988185161, 0.452298407313714,
    -0.078823449378468, -0.44069096675697,  0.165257833413781,
    -0.072783295837747, 1.133954922888981,  -1.727275138478663};

static const int test_size = 3;

Dot::Dot() {
    a_mat = b_mat = c_mat = r_mat = 0;
}

void Dot::clean_args() {
    if (a_mat)
        mkl_free(a_mat);
    if (b_mat)
        mkl_free(b_mat);
    if (c_mat)
        mkl_free(c_mat);
    if (r_mat)
        mkl_free(r_mat);
}

Dot::~Dot() {
    clean_args();
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

bool Dot::test(bool verbose) {
    clean_args();
    make_args(test_size);
    memcpy(a_mat, a_mat_test, m * k * sizeof(*a_mat));
    memcpy(b_mat, b_mat_test, k * n * sizeof(*b_mat));
    copy_args();
    compute();

    return mat_equal(r_mat, r_mat_test, m * n);
}

void Dot::print_args() {
    std::cout << "Matrix multiplication A * B." << std::endl;
    std::cout << "A =" << std::endl;
    print_mat('r', a_mat, m, k);
    std::cout << "B =" << std::endl;
    print_mat('r', b_mat, k, n);
}

void Dot::print_result() {
    std::cout << "A * B =" << std::endl;
    print_mat('r', r_mat, m, n);
}
