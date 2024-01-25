"""
Create a Global Map
===================

This script creates a map of the radial magnetic field from the CHAOS
geomagnetic field model at the core surface in 2016. The map projections are
handled by the Cartopy package, which you need to install in addition to
ChaosMagPy to execute the script.

"""

# Copyright (C) 2024 Clemens Kloss
#
# This script is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This script is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this script. If not, see <https://www.gnu.org/licenses/>.

import chaosmagpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

model = cp.CHAOS.from_mat('CHAOS-7.mat')  # load the mat-file of CHAOS-7

time = cp.mjd2000(2016, 1, 1)  # convert date to mjd2000
radius = 3485.  # radius of the core surface in km
theta = np.linspace(1., 179., 181)  # colatitude in degrees
phi = np.linspace(-180., 180, 361)  # longitude in degrees

# compute radial magnetic field from CHAOS up to degree 13
Br, _, _ = model.synth_values_tdep(time, radius, theta, phi,
                                   nmax=13, deriv=0, grid=True)

limit = 1.0  # mT colorbar limit

# create figure
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.EqualEarth()},
                       figsize=(12, 8))

fig.subplots_adjust(
    top=0.981,
    bottom=0.019,
    left=0.013,
    right=0.987,
    hspace=0.0,
    wspace=1.0
)

pc = ax.pcolormesh(phi, 90. - theta, Br/1e6, cmap='PuOr', vmin=-limit,
                   vmax=limit, transform=ccrs.PlateCarree())

ax.gridlines(linewidth=0.5, linestyle='dashed', color='grey',
             ylocs=np.linspace(-90, 90, num=7),  # parallels
             xlocs=np.linspace(-180, 180, num=13))  # meridians

ax.coastlines(linewidth=0.8, color='k')

# create colorbar
clb = plt.colorbar(pc, ax=ax, extend='both', shrink=0.8, pad=0.05,
                   orientation='horizontal')
clb.set_label('$B_r$ (mT)', fontsize=14)

plt.show()
