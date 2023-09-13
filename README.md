[![Run benchmark tests](https://github.com/IntelPython/optimizations_bench/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/IntelPython/optimizations_bench/actions/workflows/run_tests.yaml)

# Optimization Benchmarks
Collection of performance benchmarks used to present optimizations implemented for Intel(R) Distribution for Python*

## Environment Setup
To install Python environments from Intel channel along with pip-installed packages

- `conda env create -f environments/intel.yaml`
- `conda activate intel_env`

## Run tests
- `python numpy/umath/umath_mem_bench.py -v --size 10 --goal-time 0.01 --repeats 1`

## Run benchmarks
### umath
- To run python benchmarks: `python numpy/umath/umath_mem_bench.py`
- To compile and run native benchmarks (requires `icx`): `make -C numpy/umath`

### Random number generation
- To run python benchmarks: `python numpy/random/rng.py`
- To compile and run native benchmarks (requires `icx`): `make -C numpy/random`

## See also
"[Accelerating Scientific Python with Intel Optimizations](http://conference.scipy.org/proceedings/scipy2017/pdfs/oleksandr_pavlyk.pdf)" by Oleksandr Pavlyk, Denis Nagorny, Andres Guzman-Ballen, Anton Malakhov, Hai Liu, Ehsan Totoni, Todd A. Anderson, Sergey Maidanov. Proceedings of the 16th Python in Science Conference (SciPy 2017), July 10 - July 16, Austin, Texas
