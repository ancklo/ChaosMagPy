# Copyright (C) 2024 Clemens Kloss
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

"""
ChaosMagPy is a simple Python package for evaluating the CHAOS geomagnetic
field model and other models of Earth's magnetic field. The latest CHAOS model
is available at http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/.

>>> import chaosmagpy as cp
>>> chaos = cp.load_CHAOS_matfile('CHAOS-7.mat')

The following modules are available:

* chaosmagpy.chaos (functions for loading well-known field models)
* chaosmagpy.coordinate_utils (tools for transforming coordinates/components)
* chaosmagpy.model_utils (tools for evaluating field models)
* chaosmagpy.plot_utils (tools for plotting)
* chaosmagpy.data_utils (tools for reading/writing data files)
* chaosmagpy.config_utils (tools for customizing ChaosMagPy parameters)

More information on how to use ChaosMagPy can be found in the documentation
available at https://chaosmagpy.readthedocs.io/en/.

"""

__all__ = [
    "CHAOS", "load_CHAOS_matfile", "load_CHAOS_shcfile",
    "load_CovObs_txtfile", "load_gufm1_txtfile", "load_CALS7K_txtfile",
    "load_IGRF_txtfile", "basicConfig", "synth_values", "mjd2000",
    "timestamp", "dyear_to_mjd", "mjd_to_dyear"
]

__version__ = "0.14-dev"

from .chaos import (
    CHAOS,
    load_CHAOS_matfile,
    load_CHAOS_shcfile,
    load_CovObs_txtfile,
    load_gufm1_txtfile,
    load_CALS7K_txtfile,
    load_IGRF_txtfile
)

from .config_utils import basicConfig
from .model_utils import synth_values
from .data_utils import mjd2000, timestamp, dyear_to_mjd, mjd_to_dyear
