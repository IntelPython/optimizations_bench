/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"

class Cholesky : public Bench {
    public:
        Cholesky();
        ~Cholesky();
        void make_args(int size);
        void copy_args();
        void compute();

    private:
        double *x_mat, *r_mat;
        int n, lda, mat_size;
};

