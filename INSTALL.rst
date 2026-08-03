Installation
============

ChaosMagPy relies on the following (some are optional):

* python>=3.7
* numpy (>=2 if python>=3.12)
* scipy
* pandas
* h5py
* hdf5storage (>=0.2 if python>=3.12)
* pyshp>=2.3.1
* apexpy>=2.1.0 (optional, used for evaluating the ionospheric E-layer field)
* matplotlib>=3.6 (optional, used for plotting)
* lxml (optional, used for downloading latest RC-index file)

Specific installation steps for all dependencies, including optional packages,
using the conda/pip package managers are as follows:

1. Install packages with conda:

   >>> conda install python numpy scipy pandas pyshp h5py matplotlib lxml

2. Install remaining packages with pip:

   >>> pip install hdf5storage apexpy

3. Finally, install ChaosMagPy with pip:

   >>> pip install chaosmagpy

   Alternatively, if you have downloaded the distribution archives from the
   Python Package Index (PyPI) at https://pypi.org/project/chaosmagpy/#files,
   install ChaosMagPy using the built or source distributions:

   >>> pip install chaosmagpy-x.x-py3-none-any.whl

   or

   >>> pip install chaosmagpy-x.x.tar.gz

   replacing  ``x.x`` with the relevant version.
