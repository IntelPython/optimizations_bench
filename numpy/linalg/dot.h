/*
 * Copyright (C) 2016, 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"

class Dot : public Bench {
  public:
    Dot();
    ~Dot();
    void make_args(int size);
    void copy_args();
    void clean_args();
    bool test(bool verbose);
    void print_args();
    void print_result();
    void compute();

  private:
    double *a_mat, *b_mat, *c_mat, *r_mat;
    int m, n, k;
};
