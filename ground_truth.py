import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import sys, glob

plt.ion()

# ----------------------------------------------------------------------
# IMPORT YOUR FUNCTIONS
# ----------------------------------------------------------------------
from bt_functions import grab_image

# ----------------------------------------------------------------------
# LOAD BLUETIDES CATALOG
# ----------------------------------------------------------------------
catalog_fn = 'hlsp_bluetides_multi_multi_all_multi_v1_sim.csv'
catalog = Table.read(catalog_fn)

# Force the 'id' column to start at 1
catalog['id'] = 1 + np.arange(len(catalog))

if catalog.colnames[0] != 'id':
    catalog.move_column('id', 0)

TELESCOPE = 'jwst'
FILTER = 'f150w'
REDSHIFT = 7
pchi = 15

# Your global filenames list required by grab_image()
img_path = '/Users/lhenson/Documents/Research_UCR/BlueTides/jwst_all_*/hlsp_bluetides_jwst_nircam_z{redshift}*_f*w_v1_sim-*.fits'
filenames = glob.glob(img_path.format(redshift=REDSHIFT))

# ----------------------------------------------------------------------
# PREPARE STATUS FILE
# ----------------------------------------------------------------------
status_fn = catalog_fn.replace('.csv', '_status.txt')

if os.path.exists(status_fn):
    status_table = Table.read(status_fn, format='ascii')
    status = np.array(status_table['status'])
    print('Status file loaded')
else:
    status_table = Table([catalog['id'], np.zeros(len(catalog), dtype=int) - 2],
                         names=['id', 'status'])
    status = np.array(status_table['status'])
    print('New status file created')

status_dict = {
    -2: 'Not reviewed',
    -1: 'Missing data',
    0: 'No Pair',
    1: 'Pair',
}

# ----------------------------------------------------------------------
# Initialize the dynamic window — *two side-by-side images*
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# ----------------------------------------------------------------------
# MAIN REVIEW LOOP
# ----------------------------------------------------------------------
for idx, objid in enumerate(catalog['id']):

    # Skip objects already reviewed
    if status[idx] in (0, 1):
        continue

    print()
    print(f"===== Reviewing object {objid} =====")

    # ---------------------------------------------------------------
    # Load f150w raw + psf images using your grab_image()
    # ---------------------------------------------------------------
    try:
        img_nopsf, img_psf = grab_image(objid, TELESCOPE, FILTER, REDSHIFT, catalog)
    except Exception as e:
        print(f"Error loading image for {objid}: {e}")
        status[idx] = -1
        continue

    # ---------------------------------------------------------------
    # Display RAW image (no PSF)
    # ---------------------------------------------------------------
    ax[0].imshow(img_nopsf, cmap='RdGy', origin='lower')
    ax[0].set_title(f"ID {objid} — RAW ({FILTER})")
    ax[0].axis('off')

    # ---------------------------------------------------------------
    # Display PSF image
    # ---------------------------------------------------------------
    ax[1].imshow(img_psf, cmap='RdGy', origin='lower')
    ax[1].set_title(f"ID {objid} — PSF ({FILTER})")
    ax[1].axis('off')

    plt.draw()

    # ---------------------------------------------------------------
    # Ask user for classification
    # ---------------------------------------------------------------
    if "redshift" in catalog.colnames:
        print(f"z = {catalog['redshift'][idx]}")
    print(f"Current status: {status[idx]} ({status_dict[status[idx]]})")
    print("0: ok, 1: interloper, 2: artifact  (q to quit)")
    resp = input("Status: ")

    if resp == 'q':
        break
    if resp == '':
        print("Skipping (no change)")
        continue

    status[idx] = int(resp)

    # ---------------------------------------------------------------
    # Save status file after every update
    # ---------------------------------------------------------------
    status_table['status'] = status
    status_table.write(status_fn, format='ascii', overwrite=True)
    print("Status updated")

    # Clear both axes for next object
    for a in ax:
        a.cla()

# ----------------------------------------------------------------------
# Final save
# ----------------------------------------------------------------------
plt.close()
status_table['status'] = status
status_table.write(status_fn, format='ascii', overwrite=True)
print("Final status file saved")
