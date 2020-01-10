# Fast-convolution via overlap-save: a partial drop-in replacement for scipy.signal.fftconvolve

Features:

- 1D and 2D (both tested) and higher (untested) arrays
  - (Currently unsupported is convolving different-dimensional signals)
- Real and complex arrays
- Memory-mapped input/outputs fully supported (and tested)
- Supports alternative FFT engines such as PyFFTW
- Supports reflect-mode (signal assumed to reflect infinitely, instead of 0 outside its support; useful for avoiding edge effects)
- Relatively straightforward to paralellize each step of the algorithm
- Extensively unit-tested

When it can be used as a drop-in replacement for `fftconvolve`:

- when you call `fftconvolve` with `mode='same'` and `axes=None`
- [See docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html)

**Installation** `$ pip install overlap_save` then `import ols`.

**Usage**
Example (culled from unit tests):
```py
import numpy as np
from scipy.signal import fftconvolve

# Generate a 100x100 signal array and a 10x10 filter array
nx = 100
nh = 10
x = np.random.randint(-10, 10, size=(nx, nx)) + 1.0
h = np.random.randint(-10, 10, size=(nh, nh)) + 1.0

# Compute the linear convolution using the FFT, keeping the center 100x100 samples
expected = fftconvolve(x, h, mode='same')

# Use overlap-save, computing the output in 6x5-sized chunks. Instead of one huge FFT, we do a
# several tiny ones
from ols import ols
actual = ols(x, h, [6, 5])

# The two will match
assert np.allclose(expected, actual)
```

**API** `def ols(x, h, size=None, nfft=None, out=None, rfftn=None, irfftn=None, mode='constant', **kwargs)` Perform multidimensional overlap-save fast-convolution.

As mentioned in the module docstring, the output of this function will be within machine precision of `scipy.signal.fftconvolve(x, h, mode='same')`.

However, rather than computing three potentially-large FFTs (one for `x`, one for `h`, and an inverse FFT for their product), the overlap-save algorithm performs a sequence of smaller FFTs. This makes it appropriate for situations where you may not be able to store the signals' FFTs in RAM, or even cases where you cannot even store the signals themselves in RAM, i.e., when you have to memory-map them from disk.

`x` and `h` can be multidimensional (1D and 2D are extensively tested), but must have the same rank, i.e., `len(x.shape) == len(h.shape)`.

`size` is a list of integers that specifies the sizes of the output that will be computed in each iteration of the overlap-save algorithm. It must be the same length as `x.shape` and `h.shape`. If not provided, defaults to `[4 * x for x in h.shape]`, i.e., will break up the output into chunks whose size is governed by the size of `h`.

`nfft` is a list of integers that specifies the size of the FFT to be used.  It's length must be equal to the length of `size`. Each element of this list must be large enough to store the *linear* convolution, i.e., `all([nfft[i] >= size[i] + h.shape[i] - 1 for i in range(len(nfft))])` must be `True`. Set this to a multiple of small prime factors, which is the default.

If provided, the results will be stored in `out`. This is useful for memory-mapped outputs, e.g.

If not provided, `rfftn` and `irfftn` default to those in `numpy.fft`. Other implementations matching Numpy's, such as PyFFTW, will also work.

By default, `mode='constant'` assumes elements of `x` outside its boundaries are 0, which matches the textbook definition of convolution. `mode='reflect'` is also supported. It should be straightforward to add support for other modes supported by `np.pad`. Keyword arguments in `**kwargs` are passed to `np.pad`.

**Development** To simply develop this repo, instead of installing it, run the following (uses `venv` with Python 3—adapt for virtualenv if you are on Python 2):
```shell
$ git clone https://github.com/fasiha/overlap_save-py.git
$ cd overlap_save-py                  # clone the repo and change to its directory
$ git checkout -b MY-DEV-BRANCH       # check out a new git branch
$ python -m venv .                    # set up a venv (one-time)
$ source bin/activate                 # activate venv
$ pip install -r requirements.txt     # install module requirements
$ pip install -r requirements-dev.txt # install requirements for tests
$ nosetests -w .                      # run all tests. You may have to re-activate venv
```

**Changelog**

*1.1.2* support for `numpy<1.15` (where [`numpy.flip`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html) had more limited behavior)—thanks again Matteo!

*1.1.1* full complex support (thanks Matteo Bachetti! [!1](https://github.com/fasiha/overlap_save-py/pull/1) & [!2](https://github.com/fasiha/overlap_save-py/pull/2))

*1.1.0* PyFFTW and `mirror` mode added