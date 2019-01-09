/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"

class Det : public Bench {
  public:
    Det();
    ~Det();
    void make_args(int size);
    void copy_args();
    void compute();

  private:
    double *x_mat, *r_mat;
    int *ipiv;
    int m, n, lda, mn_min, mat_size;
    double result;
};
