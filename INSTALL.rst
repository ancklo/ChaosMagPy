Installation
============

ChaosMagPy relies on the following:

* python>=3.6
* numpy
* scipy
* pandas
* cython
* h5py
* hdf5storage>0.1.17
* matplotlib>=3
* cdflib (optional)
* cartopy>=0.17 (optional)
* lxml (optional)

Specific installation steps using the conda/pip package managers are as follows:

1. Install packages with conda:

   >>> conda install python numpy scipy pandas cython cartopy matplotlib h5py

2. Install packages with pip:

   >>> pip install cdflib hdf5storage lxml

3. Finally install ChaosMagPy either with pip from PyPI:

   >>> pip install chaosmagpy

   or, if you have downloaded the `package files <https://pypi.org/project/chaosmagpy/#files>`_
   into the current working directory, with:

   >>> pip install chaosmagpy-x.x-py3-none-any.whl

   or, alternatively

   >>> pip install chaosmagpy-x.x.tar.gz

   replacing ``x.x`` with the correct version.
