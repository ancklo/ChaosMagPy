Configuration
=============

ChaosMagPy internally uses a number of parameters and coefficient/data files,
whose numerical values and filepaths are stored in a dictionary-like container,
called ``basicConfig``. Usually, the content of it need not be changed.
However, if for example one wishes to compute a time series of the
external field beyond the limit of the builtin RC-index file, then ChaosMagPy
can be configured to use an updates RC-index file.

To view the parameters in ``basicConfig``, do the following:

.. code-block:: python

   import chaosmagpy as cp

   print(cp.basicConfig)

This will print a list of the parameters than can in principle be changed.
For example, it contains Earth's surface radius ``params.r_surf``, which is
used as reference radius for the spherical harmonic representation of the
magnetic potential field. For a complete list, see
:ref:`sec-configuration-utilities`.

.. _sec-configuration-change-rc-index-file:

Change RC index file
--------------------

.. note::

   Find more information about why this may be necessary in the documentation
   of :meth:`~.chaos.CHAOS.synth_coeffs_sm`.

With the latest version of the CHAOS model, it is recommended to use the latest
version of the RC-index that can be downloaded as TXT-file (``*.dat``) from the
RC website at :rc_url:`spacecenter.dk <>`, or by using the function
:func:`~.data_utils.save_RC_h5file` (locally saves the RC-index TXT-file as
HDF5-file):

.. code-block:: python

   import chaosmagpy as cp

   cp.data_utils.save_RC_h5file('my_RC_file.h5')

There is no significant difference in speed when using TXT-file or HDF5-file
formats in this case. After importing ChaosMagPy, provide the path to the new
RC-index file:

.. code-block:: python

   cp.basicConfig['file.RC_index'] = './my_RC_file.h5'

This should be done at the top of the script after the import statements,
otherwise ChaosMagPy uses the builtin RC-index file.

If you use are using an older version of CHAOS-7, and are interested in the
external field part of that model, it is recommended to use the RC file from
the archive at the bottom of the :chaos_url:`CHAOS-7 website <>` associated
with the specific version of CHAOS-7 that you are using.

Save and load custom configuration
----------------------------------

The configuration values can also be read from and written to a simple text
file in json format.

For the correct format, it is best to change the configuration parameters
during a python session and then save them to a file. For example, the
following code sets the Earth's surface radius to 6371 km (there is no reason
to do this except for the sake of this example):

.. code-block:: python

   import chaosmagpy as cp

   cp.basicConfig['params.r_surf'] = 6371

Then save the configuration dictionary to a file:

.. code-block:: python

   cp.basicConfig.save('myconfig.json')

To load this configuration file, use the following at the start of the script:

.. code-block:: python

   cp.basicConfig.load('myconfig.json')
