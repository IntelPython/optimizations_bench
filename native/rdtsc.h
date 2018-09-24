/*
 * Copyright (C) 2017 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __RDTSC_H
#define __RDTSC_H

#if defined(__ICC)
#include "immintrin.h"

typedef unsigned __int64 rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    unsigned int tmp;
    return __rdtscp(&tmp);
}

#else

#if defined(__i386__)

typedef unsigned long long int rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    rdtsc_type x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}

#elif defined(__x86_64__)

typedef unsigned long long int rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (rdtsc_type)lo)|( ((rdtsc_type)hi)<<32 );
}

#elif defined(_WIN32) || defined(_WIN64)
#include <intrin.h>

typedef unsigned __int64 rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    return __rdtsc();
}

#else

#error "THIS ARCH IS NOT SUPPORTED"

#endif

#endif

#endif /* __RDTSC_H */
