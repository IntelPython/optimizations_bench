# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import urllib.request
from os.path import join as jp

dir = 'miniconda3'
conda = jp(dir,'bin','conda')
miniconda = 'Miniconda3-latest-Linux-x86_64.sh'
miniconda_url = 'https://repo.continuum.io/miniconda/' + miniconda

if not os.path.exists(conda):
   if not os.path.exists(dir):
      os.makedirs(dir)
   if not os.path.exists(miniconda):
      urllib.request.urlretrieve(miniconda_url, miniconda)
   os.system('chmod +x %s; ./%s -b -p %s -f' % (miniconda,miniconda,dir))

if not os.path.exists(jp(dir,'envs','intel3')):
   os.system('%s create -q -y -n intel3 -c intel python=3 numpy numexpr numexpr numba scikit-learn tbb cython' % conda)

pip_env = jp(dir,'envs','pip3')
if not os.path.exists(pip_env):
   os.system('%s create -q -y -n pip3 -c intel python=3 pip llvmlite cython' % conda)
   os.system('%s/bin/pip -q install numpy scikit-learn toolz numexpr rdtsc' % pip_env)
   os.system('%s/bin/pip -q install dask numba' % pip_env)
