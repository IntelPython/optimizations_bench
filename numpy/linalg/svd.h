/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"

class SVD : public Bench {
    public:
        SVD();
        ~SVD();
        void make_args(int size);
        void copy_args();
        void compute();

    private:
        double *a_mat, *r_mat, *u_mat, *vt_mat, *s_vec;
        int n, lda, mat_size;
};
