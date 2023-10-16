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

Download a new RC-index file either directly as TXT-file (``*.dat``) from
:rc_url:`spacecenter.dk <>` or by using the function
:func:`~.data_utils.save_RC_h5file` (saves the RC-index as HDF5-file):

.. code-block:: python

   from chaosmagpy.data_utils import save_RC_h5file

   save_RC_h5file('my_RC_file.h5')

There is no significant difference in speed when using TXT-file or HDF5-file
formats in this case. After importing ChaosMagPy, provide the path to the new
RC-index file:

.. code-block:: python

   import chaosmagpy as cp

   cp.basicConfig['file.RC_index'] = './my_RC_file.h5'

This should be done at the top of the script after the import statements,
otherwise ChaosMagPy uses the builtin RC-index file.

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
