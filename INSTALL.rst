Installation
============

ChaosMagPy relies on the following (some are optional):

* python>=3.6
* numpy>=2 
* scipy
* pandas
* cython
* h5py
* hdf5storage>=0.2 (for compatibility with numpy v2.0)
* pyshp>=2.3.1
* matplotlib>=3.6 (optional, used for plotting)
* lxml (optional, used for downloading latest RC-index file)

Specific installation steps using the conda/pip package managers are as follows:

1. Install packages with conda:

   >>> conda install python "numpy>=2" scipy pandas cython pyshp h5py matplotlib lxml

2. Install remaining packages with pip:

   >>> pip install "hdf5storage>=0.2"

3. Finally install ChaosMagPy either with pip from PyPI:

   >>> pip install chaosmagpy

   Or, if you have downloaded the distribution archives from the Python Package
   Index (PyPI) at https://pypi.org/project/chaosmagpy/#files, install
   ChaosMagPy using the built distribution:

   >>> pip install chaosmagpy-x.x-py3-none-any.whl

   replacing  ``x.x`` with the relevant version, or using the source
   distribution:

   >>> pip install chaosmagpy-x.x.tar.gz
