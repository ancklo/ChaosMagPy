Installation
============

ChaosMagPy relies on the following (some are optional):

* python>=3.6
* numpy
* scipy
* pandas
* cython
* h5py
* hdf5storage>0.1.17
* matplotlib>=3
* pyshp>=2.3.1
* cdflib (optional)
* lxml (optional)

Specific installation steps using the conda/pip package managers are as follows:

1. Install packages with conda:

   >>> conda install python numpy scipy pandas cython pyshp matplotlib h5py lxml

2. Install remaining packages with pip:

   >>> pip install cdflib hdf5storage

3. Finally install ChaosMagPy either with pip from PyPI:

   >>> pip install chaosmagpy

   Or, if you have downloaded the package files from the Python Package Index
   (PyPI) at https://pypi.org/project/chaosmagpy/#files, install ChaosMagPy by
   running the command below, replacing  ``x.x`` with the specific version of
   the downloaded files:

   >>> pip install chaosmagpy-x.x-py3-none-any.whl

   or, using the compressed archive

   >>> pip install chaosmagpy-x.x.tar.gz
