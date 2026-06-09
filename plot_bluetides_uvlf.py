import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# --- BLUETIDES UVLF FOR REDSHIFT 7-9 ---

# IMPORT BLUETIDES CATALOG
catalog = Table.read('path_to_catalog/hlsp_bluetides_multi_multi_all_multi_v1_sim.csv')

# MAGNITUDE BINS 
bins_M = np.arange(-26.25, -14.75, 0.25)  # magnitude bins
bin_centers_M = 0.5 * (bins_M[:-1] + bins_M[1:])
bin_width_M = bins_M[1] - bins_M[0]

# WE CALCULATE VOLUME TWO WAYS: 
# -----------------------------
# VOLUME ASSUMING BOX OF LENGTH 400/h Mpc (ignoring cosmology)
h = cosmo.h
L_box = 400 / h     # box length
V_box = L_box**3    # box volume

# CALCULATE VOLUME ASSUMING CONSTANT ANGULAR SIZE (light cone)
z_center = 7.0
z_low = 6.5
z_high = 7.5

D_M_z7 = cosmo.comoving_transverse_distance(z_center)   # convert from physical size to angle
theta = (L_box*u.Mpc / D_M_z7) * u.rad                  # angle (box width/D_M)
area_z7 = theta**2                                      # cross-section area (square field)

print("comoving transverse distance:", D_M_z7)
print("Angular size:", theta.to(u.deg))
print("Solid angle:", area_z7.to(u.sr))

# Compute shell comoving volumes
V_inner = cosmo.comoving_volume(z_low)
V_outer = cosmo.comoving_volume(z_high)

V_shell = (area_z7 / (4 * np.pi * u.sr)) * (V_outer - V_inner)
print("Light pyramid volume:", V_shell.to(u.Mpc**3))

# REDSHIFTS TO PLOT
redshifts = [7, 8, 9]
colors = ['tab:blue', 'tab:red', 'tab:green']

fig, ax = plt.subplots()

# PLOT UVLF BY REDSHIFT
for i, z in enumerate(redshifts):
    mask = (catalog['redshift'] == z)       # mask catalog by redshiftpl
    print("redshift=", z)
    print('number of galaxies=', len(catalog[mask]))
    print()

    lum_FUV = np.log10(catalog['lum_FUV'][mask])    # UV luminosity
    M_FUV = 51.63 - 2.5 * lum_FUV                   # UV magnitude

    # UVLF using V_shell - can switch to V_box
    N_M, _ = np.histogram(M_FUV, bins=bins_M)           # binning by magnitude
    phi_M = N_M / (V_shell * bin_width_M)               # UVLF [cMpc^-3 mag^-1]
    phi_err_M = np.sqrt(N_M)/(V_shell * bin_width_M)    # Poisson error

    # plot uvlf as step function
    ax.step(
        bin_centers_M, 
        phi_M,
        where='mid', 
        label=f'Catalog UVLF z={z}',
        color=colors[i]
    )

# Points taken from Marshall et al 2022, Figure 1:
ax.scatter([-20.7, -21.4, -21.8, -22.3], [1e-4,1e-5,1e-6,1e-7], color='tab:blue',  label='Marshall+22 z=7')   
ax.scatter([-20, -21.2, -21.7, -22.2],   [1e-4,1e-5,1e-6,1e-7], color='tab:red',   label='Marshall+22 z=8')   
ax.scatter([-20, -20.9, -21.5, -21.95],  [3e-5,1e-5,1e-6,1e-7], color='tab:green', label='Marshall+22 z=9')  

ax.set_xlim(-25,-19)
ax.set_yscale('log')
ax.set_xlabel("UV Magnitude")
ax.set_ylabel(r"$\Phi$ [mag$^{-1}$ * cMpc$^{-3}$]")
ax.legend()

plt.show()
