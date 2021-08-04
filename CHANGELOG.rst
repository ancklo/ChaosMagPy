Changelog
=========

Version 0.7-dev
---------------
| **Date:** August 04, 2021
| **Release:** v0.7-dev
| **Version of CHAOS:** CHAOS-7.7 (0707)

Features
^^^^^^^^
* Added matplotlib's plot_directive for sphinx and added a gallery section
  with more examples to the documentation. 
* Added :func:`chaosmagpy.model_utils.pp_from_bspline` to convert the spline
  coefficients from B-spline to PP format.
* Changed the way piecewise polynomials are produced from the coefficients in
  shc-files. A B-spline representation is now created in an intermediate step
  to ensure coefficient time series that are smooth.
* Changed the number format to ``'16.8f'`` when writing shc-files to increase
  precision.
* Configuration parameters in ``chaosmagpy.basicConfig`` are now saved to and
  loaded from a json-formatted txt-file.
* Added keyword arguments to :meth:`chaosmagpy.chaos.CHAOS.synth_coeffs_sm`
  and :meth:`chaosmagpy.chaos.CHAOS.synth_values_sm` to provide the RC-index
  values directly instead of using the built-in RC-index file.

Version 0.6
-----------
| **Date:** March 22, 2021
| **Release:** v0.6
| **Version of CHAOS:** CHAOS-7.6 (0706), CHAOS-7.7 (0707)

News
^^^^
The latest version of CHAOS (CHAOS-7.7) corrects an error in the distributed
CHAOS-7.6 model files. The mat-file and shc-file for CHAOS-7.6 were due to a
bug identical to CHAOS-7.5, i.e. not correctly updated. The distributed spline
coefficient file for CHAOS-7.6 was correct. The CHAOS-7.7 release corrects the
errors and all CHAOS-7.7 files use updated data to March 2021.

ChaosMagPy v0.6 also works with CHAOS-7.7 and does not need to be
updated (2021-06-15).

Features
^^^^^^^^
* Added new usage sections to the documentation

Bugfixes
^^^^^^^^
* Fixed broken link to RC-index file (GitHub issue #5).
* Added lxml to installation instructions
  (needed for webpage requests, optional).
* Require hdf5storage version 0.1.17 (fixed read/write intent)

Version 0.5
-----------
| **Date:** December 23, 2020
| **Release:** v0.5
| **Version of CHAOS:** CHAOS-7.5 (0705)

Features
^^^^^^^^
* Modified "nio" colormap to be white-centered.
* Added spatial power spectrum of toroidal sources
  (:func:`chaosmagpy.model_utils.power_spectrum`)

Version 0.4
-----------
| **Date:** September 10, 2020
| **Release:** v0.4
| **Version of CHAOS:** CHAOS-7.3 (0703), CHAOS-7.4 (0704)

Features
^^^^^^^^
* Updated RC-index file to RC_1997-2020_Aug_v4.dat.
* Model name defaults to the filename it was loaded from.
* Added function to read the COV-OBS.x2 model
  (:func:`chaosmagpy.chaos.load_CovObs_txtfile`) from a text file.
* Added function to read the gufm1 model
  (:func:`chaosmagpy.chaos.load_gufm1_txtfile`) from a text file.
* Added class method to initialize :class:`chaosmagpy.chaos.BaseModel` from a
  B-spline representation.

Version 0.3
-----------
| **Date:** April 20, 2020
| **Release:** v0.3
| **Version of CHAOS:** CHAOS-7.2 (0702)

News
^^^^
The version identifier of the CHAOS model using ``x``, which stands for an
extension of the model, has been replaced in favor of a simple version
numbering. For example, ``CHAOS-6.x9`` is the 9th extension of the CHAOS-6
series. But starting with the release of the CHAOS-7 series, the format
``CHAOS-7.1`` has been adopted to indicate the first release of the series,
``CHAOS-7.2`` the second release (formerly the first extension) and so on.

Features
^^^^^^^^
* Updated RC-index file to RC_1997-2020_Feb_v4.dat.
* Removed version keyword of :class:`chaosmagpy.chaos.CHAOS` to avoid
  confusion.
* Added ``verbose`` keyword to the ``call`` method of
  :class:`chaosmagpy.chaos.CHAOS` class to avoid printing messages.
* Added :func:`chaosmagpy.data_utils.timestamp` function to convert modified
  Julian date to NumPy's datetime format.
* Added more examples to the :class:`chaosmagpy.chaos.CHAOS` methods.
* Added optional ``nmin`` and ``mmax`` to
  :func:`chaosmagpy.model_utils.design_gauss` and
  :func:`chaosmagpy.model_utils.synth_values` (nmin has been redefined).
* Added optional derivative to :func:`chaosmagpy.model_utils.colloc_matrix`
  of the B-Spline collocation.
  New implementation does not have the missing endpoint problem.
* Added ``satellite`` keyword to change default satellite names when loading
  CHAOS mat-file.

Version 0.2.1
-------------
| **Date:** November 20, 2019
| **Release:** v0.2.1
| **Version of CHAOS:** CHAOS-7.1 (0701)

Bugfixes
^^^^^^^^
* Corrected function :func:`chaosmagpy.coordinate_utils.zenith_angle` which was
  computing the solar zenith angle from ``phi`` defined as the hour angle and
  NOT the geographic longitude. The hour angle is measure positive towards West
  and negative towards East.

Version 0.2
-----------
| **Date:** October 3, 2019
| **Release:** v0.2
| **Version of CHAOS:** CHAOS-7.1 (0701)

Features
^^^^^^^^
* Updated RC-index file to recent version (August 2019, v6)
* Added option ``nmin`` to :func:`chaosmagpy.model_utils.synth_values`.
* Vectorized :func:`chaosmagpy.data_utils.mjd2000`,
  :func:`chaosmagpy.data_utils.mjd_to_dyear` and
  :func:`chaosmagpy.data_utils.dyear_to_mjd`.
* New function :func:`chaosmagpy.coordinate_utils.local_time` for a simple
  computation of the local time.
* New function :func:`chaosmagpy.coordinate_utils.zenith_angle` for computing
  the solar zenith angle.
* New function :func:`chaosmagpy.coordinate_utils.gg_to_geo` and
  :func:`chaosmagpy.coordinate_utils.geo_to_gg` for transforming geodetic and
  geocentric coordinates.
* Added keyword ``start_date`` to
  :func:`chaosmagpy.coordinate_utils.rotate_gauss_fft`
* Improved performance of :meth:`chaosmagpy.chaos.CHAOS.synth_coeffs_sm` and
  :meth:`chaosmagpy.chaos.CHAOS.synth_coeffs_gsm`.
* Automatically import :func:`chaosmagpy.model_utils.synth_values`.

Deprecations
^^^^^^^^^^^^
* Rewrote :func:`chaosmagpy.data_utils.load_matfile`: now traverses matfile
  and outputs dictionary.
* Removed ``breaks_euler`` and ``coeffs_euler`` from
  :class:`chaosmagpy.chaos.CHAOS` class
  attributes. Euler angles are now handled as :class:`chaosmagpy.chaos.Base`
  class instance.

Bugfixes
^^^^^^^^
* Fixed collocation matrix for unordered collocation sites. Endpoint now
  correctly taken into account.

Version 0.1
-----------
| **Date:** May 10, 2019
| **Release:** v0.1
| **Version of CHAOS:** CHAOS-6-x9

Features
^^^^^^^^
* New CHAOS class method :meth:`chaosmagpy.chaos.CHAOS.synth_euler_angles` to
  compute Euler angles for the satellites from the CHAOS model (used to rotate
  vectors from magnetometer frame to the satellite frame).
* Added CHAOS class methods :meth:`chaosmagpy.chaos.CHAOS.synth_values_tdep`,
  :meth:`chaosmagpy.chaos.CHAOS.synth_values_static`,
  :meth:`chaosmagpy.chaos.CHAOS.synth_values_gsm` and
  :meth:`chaosmagpy.chaos.CHAOS.synth_values_sm` for field value computation.
* RC index file now stored in HDF5 format.
* Filepaths and other parameters are now handled by a configuration dictionary
  called ``chaosmagpy.basicConfig``.
* Added extrapolation keyword to the BaseModel class
  :meth:`chaosmagpy.chaos.Base.synth_coeffs`, linear by default.
* :func:`chaosmagpy.data_utils.mjd2000` now also accepts datetime class
  instances.
* :func:`chaosmagpy.data_utils.load_RC_datfile` downloads latest RC-index file
  from the website if no file is given.

Bugfixes
^^^^^^^^
* Resolved issue in :func:`chaosmagpy.model_utils.degree_correlation`.
* Changed the date conversion to include hours and seconds not just the day
  when plotting the timeseries.

Version 0.1a3
-------------
| **Date:** February 19, 2019
| **Release:** v0.1a3

Features
^^^^^^^^
* New CHAOS class method :meth:`chaosmagpy.chaos.CHAOS.save_matfile` to output
  MATLAB compatible files of the CHAOS model (using the ``hdf5storage``
  package).
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
* :func:`chaosmagpy.data_utils.mjd_to_dyear` and
  :func:`chaosmagpy.data_utils.dyear_to_mjd` convert time with microseconds
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
