"""
Spatial Power Spectra
=====================

This script creates a plot of the spatial power spectrum of the time-dependent
internal field from the CHAOS geomagnetic field model.

"""

import chaosmagpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

model = cp.CHAOS.from_mat('CHAOS-7.mat')  # load the mat-file of CHAOS-7

nmax = 20
time = cp.data_utils.mjd2000(2018, 1, 1)
degrees = np.arange(1, nmax+1, dtype=int)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

for deriv, label in enumerate(['nT', 'nT/yr', 'nT/yr$^2$']):

    # get spatial power spectrum from time-dependent internal field in CHAOS
    spec = model.model_tdep.power_spectrum(time, nmax=nmax, deriv=deriv)
    ax.semilogy(degrees, spec, label=label)

ax.legend()
ax.grid(which='both')

ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax.set_title("Spatial power spectra at Earth's surface", fontsize=14)
ax.set_xlabel('Spherical harmonic degree')

plt.tight_layout()
plt.show()
