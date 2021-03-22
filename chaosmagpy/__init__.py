from chaosmagpy.chaos import (CHAOS, load_CHAOS_matfile, load_CHAOS_shcfile,
                              load_CovObs_txtfile, load_gufm1_txtfile)
from chaosmagpy.config_utils import basicConfig
from chaosmagpy.model_utils import synth_values


__all__ = ["CHAOS", "load_CHAOS_matfile", "load_CHAOS_shcfile",
           "load_CovObs_txtfile", "load_gufm1_txtfile",
           "basicConfig", "synth_values"]

__version__ = "0.6"
