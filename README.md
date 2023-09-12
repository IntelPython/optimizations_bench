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

## See also
"[Accelerating Scientific Python with Intel Optimizations](http://conference.scipy.org/proceedings/scipy2017/pdfs/oleksandr_pavlyk.pdf)" by Oleksandr Pavlyk, Denis Nagorny, Andres Guzman-Ballen, Anton Malakhov, Hai Liu, Ehsan Totoni, Todd A. Anderson, Sergey Maidanov. Proceedings of the 16th Python in Science Conference (SciPy 2017), July 10 - July 16, Austin, Texas
