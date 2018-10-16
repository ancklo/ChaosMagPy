Installation
============

ChaosMagPy relies on the following:

* python>=3.6
* numpy
* scipy
* pandas
* cython
* cartopy
* matplotlib<3
* cdflib

Specific installation steps using the conda/pip package managers are as follows:

1. Install packages with conda:

   >>> conda install python numpy scipy pandas cython cartopy matplotlib=2

   (At the moment, you need matplotlib 2 as pcolormesh is not working with the
   current cartopy release. But it will be hopefully updated soon.)

2. Install cdflib with pip:

   >>> pip install cdflib

3. Finally install ChaosMagPy either with pip from PyPi:

   >>> pip install chaosmagpy

   or, if you have downloaded the `package files <https://pypi.org/project/chaosmagpy/#files>`_
   to the current working directory, with:

   >>> pip install chaosmagpy-x.x-py3-none-any.whl

   or, alternatively

   >>> pip install chaosmagpy-x.x.tar.gz

   replacing ``x.x`` with the correct version.
