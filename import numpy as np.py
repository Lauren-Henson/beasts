import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# --- BLUETIDES UVLF FOR REDSHIFT 7-9 ---

# IMPORT BLUETIDES CATALOG
catalog = Table.read('/Users/lhenson/Documents/Research_UCR/BlueTides/hlsp_bluetides_multi_multi_all_multi_v1_sim.csv')

# MAGNITUDE BINS 
bins_M = np.arange(-26.25, -14.75, 0.25)  # magnitude bins
bin_centers_M = 0.5 * (bins_M[:-1] + bins_M[1:])
bin_width_M = bins_M[1] - bins_M[0]

# VOLUME
h = 0.7
L_box = 400 / h
V_box = L_box**3

# COMOVING DISTANCES FOR z=7,8,9
D_C = [358.3, 300.4, 256.5]

# COLORS FOR PLOTTING
colors = ['tab:blue', 'tab:red', 'tab:green']

fig, axes = plt.subplots(1,2,figsize=(12,5),sharex=True)

for i, z in enumerate([7, 8, 9]):
    mask = (catalog['redshift'] == z) #& (catalog['lum_FUV'] > 4.92 * 10**28.5)
    lum_FUV = np.log10(catalog['lum_FUV'][mask])
    M_FUV = 51.63 - 2.5 * lum_FUV

    M = 51.63 - 2.5 * lum_FUV #+ 2.5 * np.log10(1+7)
    print(len(M))

    print('redshift=', z)
    print('==================')

    # UVLF
    N_M, _ = np.histogram(M, bins=bins_M)
    phi_M = N_M / (V_box * bin_width_M)
    print(i,z)
    print(bins_M)
    print(bin_width_M)
    print(phi_M)
    print('------------')
    # phi_M = phi_M / 5
    phi_err_M = np.sqrt(N_M)/(V_box * bin_width_M)

    N_z = N_M * D_C[i] / (L_box * bin_width_M)
    # N_z = N_z / 10

    # # finding values to correct
    # for t in targets_UVLF:
    #     idx = np.nanargmin(np.abs(phi_M.value - t))
    #     print(f"{t:.0e}:")
    #     print(f"   Closest phi = {phi_M[idx]:.3e}")
    #     print(f"   M = {bin_centers_M[idx]:.3f}")
    # print('---------------------')
    # for t in targets_N:
    #     idx = np.nanargmin(np.abs(N_z - t))
    #     print(f"{t:.0e}:")
    #     print(f"   N_z = {N_z[idx]:.3e}")
    #     print(f"   M = {bin_centers_M[idx]:.3f}")

    # plot uvlf
    axes[0].step(
        bin_centers_M, 
        phi_M,
        where='mid', 
        label=f'Catalog UVLF z={z}',
        color=colors[i]
        )

    # plot number density
    axes[1].step(
        bin_centers_M, 
        N_z,
        where='mid', 
        label=f'Number Density z={z}',
        color=colors[i]
        )

print(V_box)
print(V_shell)

axes[0].scatter([-20.7, -21.4, -21.8, -22.3], [1e-4,1e-5,1e-6,1e-7], color='tab:blue')    # z=7
axes[0].scatter([-20, -21.2, -21.7, -22.2],   [1e-4,1e-5,1e-6,1e-7], color='tab:red')    # z=8
axes[0].scatter([-20, -20.9, -21.5, -21.95],  [3e-5,1e-5,1e-6,1e-7], color='tab:green')         # z=9

# linear fit to UVLF z=7
x = np.array([-21.375, -22.625, -23.125, -24.625])
y = np.array([-20.7, -21.4, -21.8, -22.3])
axes[0].scatter(f(x,y), [1e-4,1e-5,1e-6,1e-7], color='k')

axes[0].set_xlim(-25,-19)
axes[0].set_yscale('log')
axes[0].set_ylabel(r"$\Phi$")
axes[0].legend()

axes[1].scatter([-20.8, -21.4, -21.8, -22.2, -22.4], [1e3,1e2,1e1,1e0,1e-1], color='tab:blue')    # z=7
axes[1].scatter([-19.9, -21.3, -21.7, -22.1, -22.1], [1e3,1e2,1e1,1e0,1e-1], color='tab:red')    # z=8
axes[1].scatter([-20.5, -21.4, -22.1, -22.4],        [1e2,1e1,1e0,1e-1], color='tab:green')        # z=9

# linear fit to N_z z=7
x = np.array([-21.875, -22.875, -23.375, -26.125, -26.125])
y = np.array([-20.8, -21.4, -21.8, -22.2, -22.4])
axes[1].scatter(f(x,y), [1e3, 1e2,1e1,1e0,1e-1], color='k')

axes[1].set_ylabel(r"N/mag/deg^2")
axes[1].set_yscale('log')