[![Build Status](https://travis-ci.org/IntelPython/optimizations_bench.svg?branch=master)](https://travis-ci.org/IntelPython/optimizations_bench)

# Optimization Benchmarks
Collection of performance benchmarks used to present optimizations implemented for Intel(R) Distribution for Python*

## Environment Setup
To install Python environments from Intel channel along with pip-installed packages

- `python3 create_python_envs.py`
        
## Run tests
- `python3 run_tests.py`

## Run benchmarks
### umath
- To run python benchmarks: `python numpy/umath/umath_mem_bench.py`
- To compile and run native benchmarks (requires icc): `make -C numpy/umath`

### Random number generation
- To run python benchmarks: `python numpy/random/rng.py`
- To compile and run native benchmarks (requires icc): `make -C numpy/random`
