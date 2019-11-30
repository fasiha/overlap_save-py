from setuptools import setup

setup(
    name='overlap_save',
    version='1.1.0',
    author='Ahmed Fasih',
    author_email='ahmed@aldebrn.me',
    description="Overlap-save method for fast convolution, like Scipy's fftconvolve, but memory-efficient",
    license='Unlicense',
    url='https://github.com/fasiha/overlap_save-py',
    py_modules=['ols'],
    install_requires=['array-range', 'nextprod', 'numpy'],
    test_suite='nose.collector',
    tests_require=['nose', 'scipy', 'pyfftw'],
    zip_safe=True,
    keywords='overlap save add fast convolution convolve fft fftconvolve fir filter',
)
