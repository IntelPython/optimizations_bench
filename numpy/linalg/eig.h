/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"
#include <complex.h>

class Eig : public Bench {
  public:
    Eig();
    ~Eig();
    void make_args(int size);
    void copy_args();
    void compute();

  private:
    double *a_mat, *r_mat, *vl_mat, *vr_mat, *wr_vec, *wi_vec;
    double _Complex *vr_mat_complex, *w_vec_complex;
    int n, lda, ldvl, ldvr, mat_size;
    bool only_real;
};
