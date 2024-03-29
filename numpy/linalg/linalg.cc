/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "cholesky.h"
#include "det.h"
#include "dot.h"
#include "eig.h"
#include "inv.h"
#include "lu.h"
#include "qr.h"
#include "svd.h"

#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <map>
#include <vector>

static const struct option longopts[] = {
    {"size", required_argument, nullptr, 'n'},
    {"reps", required_argument, nullptr, 'r'},
    {"samples", required_argument, nullptr, 's'},
    {"prefix", required_argument, nullptr, 'p'},
    {"verbose", no_argument, nullptr, 'v'},
    {"help", no_argument, nullptr, 'h'},
    {"test", no_argument, nullptr, 't'},
    {0, 0, 0, 0}};

int main(int argc, char *argv[]) {

    std::map<std::string, Bench *> all_benches = {{"cholesky", new Cholesky()},
                                                  {"det", new Det()},
                                                  {"dot", new Dot()},
                                                  {"eig", new Eig()},
                                                  {"inv", new Inv()},
                                                  {"lu", new LU()},
                                                  {"qr", new QR()},
                                                  {"svd", new SVD()}};

    int n = 1000;
    int reps = 3;
    int samples = 1;
    bool verbose = false;
    bool test = false;
    std::string prefix = "Native-C";

    int intarg;
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "vthn:r:s:p:", longopts,
                              &option_index)) != -1) {
        switch (opt) {
        case 'n':
        case 'r':
        case 's':
            try {
                intarg = std::stoi(optarg);
            } catch (const std::exception &ex) {
                std::cerr << "error: could not convert number in args: ";
                std::cerr << ex.what() << std::endl;
                return EXIT_FAILURE;
            }

            if (intarg < 1) {
                std::cerr << "error: non-positive integer argument: ";
                std::cerr << optarg << std::endl;
                return EXIT_FAILURE;
            }
            break;
        case 'p':
            prefix = optarg;
            break;
        case 'v':
            verbose = true;
            break;
        case 't':
            test = true;
            break;
        case 'h':
            std::cout << "usage: " << argv[0] << " [-h] [-t] [-v]";
            std::cout << " [-n SIZE] [-r REPETITIONS] [-s SAMPLES]";
            std::cout << " [BENCHMARKS...]" << std::endl;
            return EXIT_SUCCESS;
        case '?':
        default:
            return EXIT_FAILURE;
        }

        switch (opt) {
        case 'n':
            n = intarg;
            break;
        case 'r':
            reps = intarg;
            break;
        case 's':
            samples = intarg;
            break;
        }
    }

    std::vector<std::string> benches;
    if (optind < argc) {
        for (; optind < argc; optind++)
            benches.push_back(argv[optind]);
    } else {
        // execute all by default
        for (auto const &bench : all_benches) {
            benches.push_back(bench.first);
        }
    }

    if (!test)
        std::cout << "Prefix,Function,Size,Time" << std::endl;

    int return_value = 0;

    for (auto const &bench : benches) {
        if (all_benches.count(bench) == 0) {
            std::cerr << "# Ignoring invalid bench name: " << bench
                      << std::endl;
            continue;
        }

        Bench *real_bench = all_benches[bench];

        if (test) {
            if (verbose)
                std::cout << "---" << std::endl;

            if (!real_bench->test(verbose)) {
                std::cout << "FAIL: " << bench;
                return_value = 1;
            } else {
                std::cout << "pass: " << bench;
            }
            std::cout << std::endl;

            if (verbose) {
                real_bench->print_args();
                real_bench->print_result();
            }
            continue;
        }
        real_bench->make_args(n);

        if (verbose)
            real_bench->print_args();

        // warm up
        real_bench->copy_args();
        real_bench->compute();

        for (int i = 0; i < samples; i++) {

            std::chrono::duration<double> timedelta(0.0);

            for (int j = 0; j < reps; j++) {
                real_bench->copy_args();
                auto t0 = std::chrono::system_clock::now();
                real_bench->compute();
                auto t1 = std::chrono::system_clock::now();
                timedelta += t1 - t0;
            }

            std::cout << prefix << ",";
            std::cout << bench << ",";
            std::cout << n << ",";
            std::cout << (double) timedelta.count() / reps;
            std::cout << std::endl;
        }

        if (verbose)
            real_bench->print_result();
    }

    // Free benches allocated in heap
    for (auto const &bench : all_benches) {
        delete all_benches[bench.first];
    }

    return return_value;
}
