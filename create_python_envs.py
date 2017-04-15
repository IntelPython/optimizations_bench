# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
   os.system('%s/bin/pip -q install numpy scikit-learn toolz numexpr rdtsc timer' % pip_env)
   os.system('%s/bin/pip -q install dask numba' % pip_env)
