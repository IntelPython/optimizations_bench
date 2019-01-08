/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"

class QR : public Bench {
    public:
        QR();
        ~QR();
        void make_args(int size);
        void copy_args();
        void compute();

    private:
        double *x_mat_init, *x_mat, *r_mat, *tau_vec;
        int n, lda, mat_size;
};
