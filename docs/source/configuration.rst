Configuration
=============

ChaosMagPy internally uses a number of parameters and coefficient/data files,
whose numerical values and filepaths are stored in a dictionary-like container,
called ``basicConfig``. Usually, the content of it need not be changed.
However, if for example one wishes to compute a time series of the
external field beyond the limit of the builtin RC index file, then ChaosMagPy
can be configured to use an updates RC index file.

To view the parameters in ``basicConfig``, do the following:

.. code-block:: python

   import chaosmagpy as cp

   print(cp.basicConfig)

This will print a list of the parameters than can in principle be changed.
For example, it contains Earth's surface radius ``params.r_surf``, which is
used as reference radius for the spherical harmonic representation of the
magnetic potential field. For a complete list, see
:ref:`label-configuration-list`.

Change RC index file
--------------------

Download a new RC-index file either directly as ``dat``-file from
`spacecenter.dk <http://www.spacecenter.dk/files/magnetic-models/RC/current/>`_
or using the function :func:`data_utils.save_RC_h5file` (saved as
``h5``-file):

.. code-block:: python

   from chaosmagpy.data_utils import save_RC_h5file

   save_RC_h5file('my_RC_file.h5')

There is no difference in speed when using ``dat`` or ``h5`` file format in
this case. After importing ChaosMagPy, provide the path to the new RC-index
file:

.. code-block:: python

   import chaosmagpy as cp

   cp.basicConfig['file.RC_index'] = 'my_RC_file.h5'

This should be done somewhere at the beginning of the script, otherwise
ChaosMagPy uses the builtin RC index file.

Save and load custom configuration
----------------------------------

The configuration values can also be read from and written to a simple text
file. To ensure the correct format of the text file, it is best to save the
current configuration:

.. code-block:: python

   import chaosmagpy as cp

   cp.basicConfig.save('my_config.txt')

A typical line looks like this:

.. code-block:: bash

   params.r_surf : 6371.2

Comments have to start with ``#``, empty lines are skipped and key-value
pairs are separated with ``:``. Change the values in the file and load it into
ChaosMagPy at the beginning of the script:

.. code-block:: python

   cp.basicConfig.load('my_config.txt')
