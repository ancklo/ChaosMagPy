Changelog
=========

Version 0.2
-------------
| **Date:** August 6, 2019
| **Release:** v0.2

Features
^^^^^^^^
* New function ``gg_to_geo`` and  ``geo_to_gg`` for transforming geodetic and
  geocentric coordinates.
* Added keyword ``start_date`` to ``rotate_gauss_fft``
* Improved performance of ``synth_coeffs_sm`` and ``synth_coeffs_gsm``.
* Automatically import ``model_utils.synth_values``.

Deprecations
^^^^^^^^^^^^
* Removed ``breaks_euler`` and ``coeffs_euler`` from ``CHAOS`` class
  attributes. Euler angles are now handled as ``Base`` class instance.

Version 0.1
-------------
| **Date:** May 10, 2019
| **Release:** v0.1

Features
^^^^^^^^
* New CHAOS class method ``synth_euler_angles`` to compute euler angles for
  the satellites from the CHAOS model (used to rotate vectors from
  magnetometer frame to the satellite frame).
* Added CHAOS class methods ``synth_values_tdep``, ``synth_values_static``,
  ``synth_values_gsm`` and ``synth_values_sm`` for field value computation.
* RC index file now stored in HDF5 format.
* Filepaths and other parameters are now handled by ``basicConfig``.
* Added extrapolation keyword to CHAOS class method ``synth_coeffs``, linear by
  default.
* ``mjd2000`` now also accepts datetime class instances.
* ``load_RC_datfile`` downloads latest RC-index file from the website if no
  file is given.

Bugfixes
^^^^^^^^
* Resolved issue in ``model_utils.degree_correlation``.
* Changed the date conversion to include hours and seconds not just the day
  when plotting the timeseries.

Version 0.1a3
-------------
| **Date:** February 19, 2019
| **Release:** v0.1a3

Features
^^^^^^^^
* New CHAOS class method ``save_matfile`` to output MATLAB compatible
  files of the CHAOS model (using the ``hdf5storage`` package).
* Added ``epoch`` keyword to basevector input arguments of GSM, SM and MAG
  coordinate systems.

Bugfixes
^^^^^^^^
* Fixed problem of the setup configuration for ``pip`` which caused importing
  the package to fail although installation was indicated as successful.

Version 0.1a2
-------------
| **Date:** January 26, 2019
| **Release:** v0.1a2

Features
^^^^^^^^
* ``mjd_to_dyear`` and ``dyear_to_mjd`` convert time with microseconds
  precision to prevent round-off errors in seconds.
* Time conversion now uses built-in ``calendar`` module to identify leap year.

Bugfixes
^^^^^^^^
* Fixed wrong package requirement that caused the installation of
  ChaosMagPy v0.1a1 to fail with ``pip``. If installation of v0.1a1 is needed,
  use ``pip install --no-deps chaosmagpy==0.1a1`` to ignore faulty
  requirements.


Version 0.1a1
-------------
| **Date:** January 5, 2019
| **Release:** v0.1a1

Features
^^^^^^^^
* Package now supports Matplotlib v3 and Cartopy v0.17.
* Loading shc-file now converts decimal year to ``mjd2000`` taking leap years
  into account by default.
* Moved ``mjd2000`` from ``coordinate_utils`` to ``data_utils``.
* Added function to compute degree correlation.
* Added functions to compute and plot the power spectrum.
* Added flexibility to the function synth_values: now supports NumPy
  broadcasting rules.
* Fixed CHAOS class method synth_coeffs_sm default source parameter: now
  defaults to ``'external'``.

Deprecations
^^^^^^^^^^^^
* Optional argument ``source`` when saving shc-file has been renamed to
  ``model``.
* ``plot_external_map`` has been renamed to ``plot_maps_external``
* ``synth_sm_field`` has been renamed to ``synth_coeffs_sm``
* ``synth_gsm_field`` has been renamed to ``synth_coeffs_gsm``
* ``plot_static_map`` has been renamed to ``plot_maps_static``
* ``synth_static_field`` has been renamed to ``synth_coeffs_static``
* ``plot_tdep_maps`` has been renamed to ``plot_maps_tdep``
* ``synth_tdep_field`` has been renamed to ``synth_coeffs_tdep``


Version 0.1a0
-------------
| **Date:** October 13, 2018
| **Release:** v0.1a0

Initial release to the users for testing.
