Installation
============

ChaosMagpy relies on the following packages:

* numpy
* scipy
* pandas
* cython
* cartopy
* matplotlib<3
* cdflib

Specific installation steps using the conda package manager are as follows:

1. Install packages with conda:

   >>> conda install python numpy scipy pandas cython cartopy matplotlib=2

   (At the moment, you need matplotlib 2 as pcolormesh is not working with the
   current cartopy release. But it will be hopefully updated soon.)

2. Install cdflib with pip:

   >>> pip install cdflib

3. Finally install ChaosMagPy either with pip from PyPi:

   >>> pip install chaosmagpy

   or, if you downloaded the `package files <https://pypi.org/project/chaosmagpy/#files>`_
   in the current working directory, with:

   >>> pip install chaosmagpy-x.x-py3-none-any.whl

   or, alternatively

   >>> pip install chaosmagpy-x.x.tar.gz

   replacing ``x.x`` with the correct version.
