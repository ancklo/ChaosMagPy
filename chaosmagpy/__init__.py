"""
`chaosmagpy` is a simple Python package for evaluating the CHAOS geomagnetic
field model and other models of Earth's magnetic field. The CHAOS-7 model can
be downloaded at `http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/`.

>>> import chaosmagpy as cp
>>> chaos = cp.load_CHAOS_matfile('CHAOS-7.mat')

The following modules are available:

* chaosmagpy.chaos
* chaosmagpy.coordinate_utils
* chaosmagpy.model_utils
* chaosmagpy.plot_utils
* chaosmagpy.data_utils
* chaosmagpy.config_utils

More information on how to use ChaosMagPy can be found in the documentation
available at `https://chaosmagpy.readthedocs.io/en/master/`.

"""

__all__ = [
    "CHAOS", "load_CHAOS_matfile", "load_CHAOS_shcfile",
    "load_CovObs_txtfile", "load_gufm1_txtfile", "load_CALS7K_txtfile",
    "basicConfig", "synth_values", "mjd2000", "timestamp", "dyear_to_mjd",
    "mjd_to_dyear"
]

__version__ = "0.13-dev"

from .chaos import (
    CHAOS,
    load_CHAOS_matfile,
    load_CHAOS_shcfile,
    load_CovObs_txtfile,
    load_gufm1_txtfile,
    load_CALS7K_txtfile
)

from .config_utils import basicConfig
from .model_utils import synth_values
from .data_utils import mjd2000, timestamp, dyear_to_mjd, mjd_to_dyear
