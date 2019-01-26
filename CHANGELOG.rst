Changelog
=========

Version 0.1a2
-------------
| **Date:** January 26, 2019
| **Release:** v0.1a2

Features
^^^^^^^^
* ``mjd_to_dyear`` and ``dyear_to_mjd`` convert time with microseconds
  precision to prevent round-off errors in seconds.
* Time conversion now uses built-in ``calender`` module to identify leap year.

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
