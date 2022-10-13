API References
==============

The ChaosMagPy package consists of several modules, which contain classes and
functions that are relevant in geomagnetic field modelling:

* **chaosmagpy.chaos**: Everything related to loading specific geomagnetic
  field models, such as CHAOS, Cov-Obs, and other piecewise-polynomial
  spherical harmonic models (see Sect. `Core Functionality`_).
* **chaosmagpy.coordinate_utils**: Everything related to coordinate
  transformations and change of reference frames
  (see Sect. `Coordinate Transformations`_).
* **chaosmagpy.model_utils**: Everything related to evaluating geomagnetic
  field models, which includes functions for evaluating B-splines, Legendre
  polynomials and spherical harmonics (see Sect. `Model Utilities`_)
* **chaosmagpy.plot_utils**: Everything related to plotting geomagnetic field
  model outputs, i.e. plotting of timeseries, maps and spatial power
  spectra (see Sect. `Plotting Utilities`_).
* **chaosmagpy.data_utils**: Everything related to loading and saving datasets
  and model coefficients from and to files. This also includes functions for
  transforming datetime formats (see Sect. `Data Utilities`_).
* **chaosmagpy.config_utils**: Everything related to the configuration of
  ChaosMagPy, which includes setting parameters and paths to builtin data files
  (see Sect. `Configuration Utilities`_).

Core Functionality
------------------

.. automodule:: chaosmagpy.chaos
    :noindex:
    :no-members:

Coordinate Transformations
--------------------------

.. automodule:: chaosmagpy.coordinate_utils
    :noindex:
    :no-members:

Model Utilities
---------------

.. automodule:: chaosmagpy.model_utils
    :noindex:
    :no-members:

Plotting Utilities
------------------

.. automodule:: chaosmagpy.plot_utils
    :noindex:
    :no-members:

Data Utilities
--------------

.. automodule:: chaosmagpy.data_utils
    :noindex:
    :no-members:

.. _sec-configuration-utilities:

Configuration Utilities
-----------------------

.. automodule:: chaosmagpy.config_utils
    :noindex:
    :no-members:
