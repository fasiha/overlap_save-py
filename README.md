# Multidimensional overlap-save method for fast-convolution

Features:

- 1D and 2D (both tested) and higher (untested) arrays
  - (Currently unsupported is convolving different-dimensional signals)
- Memory-mapped input/outputs fully supported (and tested)
- Relatively straightforward to paralellize each step of the algorithm
- Real-only, but readily modifiable to complex (convert `rfft` to `fft`)
- Equivalent to a `np.roll` of `scipy.signal.fftconvolve(mode='same')` (this is tested)
- Extensively unit-tested

The semantics of the convolution are as follows (the following is culled from
the unit test):
```py
import numpy as np

# Generate a 100x100 signal array and a 10x10 filter array
nx = 100
nh = 10
x = np.random.randint(-10, 10, size=(nx, nx)) + 1.0
h = np.random.randint(-10, 10, size=(nh, nh)) + 1.0

# Compute the linear convolution using the FFT, keep the first 100x100 samples
ngold = np.array(x.shape) + np.array(h.shape) - 1
expected = np.real(np.fft.ifft2(np.fft.fft2(x, ngold) *
                                np.conj(np.fft.fft2(h, ngold))))[:x.shape[0], :x.shape[1]]

# Use overlap-save, computing the output in 6x5-sized chunks. Instead of one
# huge FFT, we do a sequence of tiny ones
from ols import ols
actual = ols(x, h, [6, 5])

# The two will match
assert np.allclose(expected, actual)
```

Therefore, if you're computing fast-convolution as an IFFT of the product of FFTs, this module can
function as a drop-in replacement.

If you're using `scipy.signal.fftconvolve()` with `mode='same'`, then you have to roll the output
of this module to match what you have. You'll also have to throw away some data at the end due to
edge effects. Again culled from the unit-test:
```py
from scipy.signal import fftconvolve
conv = fftconvolve(x, h[::-1, ::-1], mode='same')
conv = np.roll(conv, [-(nh // 2)] * 2, [-1, -2])
assert np.allclose(conv[:-(nh // 2), :-(nh // 2)], gold[:-(nh // 2), :-(nh // 2)])
```
