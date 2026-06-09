import glob
import corner
import sep

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import pandas as pd
import numpy as np

from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.nddata import block_reduce

from photutils.detection import find_peaks


redshift = 7

catalog = Table.read('hlsp_bluetides_multi_multi_all_multi_v1_sim.csv')
if 'id' in catalog.colnames:
    catalog.replace_column('id', 1 + np.arange(len(catalog)))
else:
    catalog.add_column(1 + np.arange(len(catalog)), name='id', index=0)
catalog.write('og_catalog.ecsv', overwrite=True)



def grab_image(objid,telescope,filter,redshift,catalog):
    ''' Grabs image based on user-inputed objid and filter
        Assume redshift = 7
        Returns image data with and without the psf
    '''
    if telescope == 'jwst':
        img_path = f'/Users/lhenson/Documents/Research_UCR/BlueTides/jwst_all_*/hlsp_bluetides_jwst_nircam_z{redshift}*_f*w_v1_sim-*.fits'
    elif telescope == 'euclid':
        img_path = f'/Volumes/SeagateBack/bluetides/euclid_*/hlsp_bluetides_euclid_nisp_z{redshift}-f*_v1_sim-*.fits'
    else:
        print("unrecognized telescope")

    filenames = glob.glob(img_path)       # finds all files in path with given format 

    idx = list(catalog['id']).index(objid)
    
    fileNumber = catalog['fileNumber'][idx]           # selects column and row of catalog based on fileNumber
    extNumber = catalog['extensionNumber'][idx]       # selects column and row of catalog based on extensionNumber

    if redshift == 7:
        # hsel dimsension is objid x filenames (default 10 x 56)
        # arrary of booleans to see if there's a file present for a particular catalog object
        hsel = [(f'file{fileNumber}' in f) & (f'_{filter}' in f) & ('_psf' in f) for f in filenames]      
        hsel_no_psf = [(f'file{fileNumber}' in f) & (f'_{filter}' in f) & ('_nopsf' in f) for f in filenames]
        print(fileNumber,filter)
        print(filenames)
    else:
        hsel = [(filter in f) & ('_psf' in f) for f in filenames]
        hsel_no_psf = [(filter in f) & ('_nopsf' in f) for f in filenames]
        print('file found')
    if not any(hsel):
        print(f'No file found for {filter}')

    #fn selects files from filenames using hsel, the [0] converts this from an array element back to a string
    fn = np.array(filenames)[hsel][0]       
    fn_no_psf = np.array(filenames)[hsel_no_psf][0]

    hdu = fits.open(fn)[extNumber]
    hdu_no_psf = fits.open(fn_no_psf)[extNumber]

    img_nopsf = hdu_no_psf.data
    img = hdu.data

    return(img_nopsf,img)


def reduce(image, telescope, filter):
    ''' 
    Block reduction performed on Euclid NISP image.
    Converts Bluetides pixel scale (0.15"/pix) to
    native Euclid NISP pixel scale (0.3"/pix).
    '''
    if telescope == 'jwst':
        if filter in ['f090w', 'f115w', 'f150w', 'f200w']:
            # SW is 0.031
            img = block_reduce(image, (4, 4), np.sum)
        else:
            # LW is 0.063
            img = block_reduce(image, (2, 2), np.sum)
        return img
    
    if telescope == 'euclid':
        if filter.lower() in ['y', 'j', 'h', 'nisp-y', 'nisp-j', 'nisp-h']:
            # 0.15" → 0.3"  ==> factor of 2
            img = block_reduce(image, (2, 2), np.sum)
        else:
            raise ValueError(f"Unknown or unsupported Euclid NISP filter: {filter}")
        return img
    
    else:
        print("Unrecognized telescope. Options are jwst or euclid")


def add_noise(image,filt,noise_level=0):
    image = image.copy()
    
    if filt in ['f090w', 'f115w', 'f150w', 'f200w']:
        reduction_factor = 2
        raw_noise = np.random.normal(0, 1, (36,36))
    else:
        reduction_factor = 3
        raw_noise = np.random.normal(0, 1, (11,11))

    noise_kernel_size = 0.75 
    conv_noise = convolve_fft(raw_noise, Gaussian2DKernel(noise_kernel_size)) 
    
    image = block_reduce(image, (reduction_factor, reduction_factor), np.sum)
    image += conv_noise*noise_level 

    return image


def plot_object(objid, telescope, filter='all',catalog=catalog,redshift=7,add_noise=False):
    ''' Plots galaxies of user-inputed objid and filter
        Filter options: 'all' or any single one of the filters listed below:
        'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w'
        Returns raw image and with psf
    '''
    #===========================================#
    
    if telescope == 'jwst':
        filters =  ['f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
    elif telescope == 'euclid':
        filters = ['h', 'y', 'j']
    else:
        print('telescope not recognized, options are "jwst" or "euclid"')
    pchi = 10

    if filter in filters: 
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
        img_nopsf, img= grab_image(objid,telescope,filter,redshift=redshift,catalog=catalog)

        ax = axes[0]

        ax.imshow(img_nopsf, cmap='RdGy', origin='lower', norm=SymLogNorm(0.5, 1, vmin=-pchi, vmax=pchi))
        ax.set_title(f'{filter} - Raw Image')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # JWST NIRSpec resolution
        ax = axes[1]

        # resample to 0.1" pixels, ROUGHLY!
        # img = reduce(img,filter)

        if add_noise:
            img = add_noise(img,filt=filter,noise_level=0.5)
    
        ax.imshow(img, cmap='RdGy', origin='lower', norm=SymLogNorm(0.5, 1, vmin=-pchi, vmax=pchi))
        ax.set_title(f'{filter} - With PSF')
        ax.set_xticks([])
        ax.set_yticks([])
        # axes[0].set_ylabel(f'Object {objid}')

    elif filter == 'all':
        fig, axes = plt.subplots(nrows=2, ncols=len(filters), figsize=(3*len(filters), 6))

        for i, filter in enumerate(filters):

            img_nopsf, img = grab_image(objid,telescope,filter,redshift=redshift,catalog=catalog)
           
            ax = axes[0,i]

            ax.imshow(img_nopsf, cmap='RdGy', origin='lower', norm=SymLogNorm(0.5, 1, vmin=-pchi, vmax=pchi))
            ax.set_title(f'{filter} - No PSF',fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # JWST NIRSpec resolution
            ax = axes[1,i]
    
            # resample to 0.1" pixels, ROUGHLY!
            # img = reduce(img,filter)

            if add_noise:
                img = add_noise(img,filt=filter,noise_level=0.5)
        
            ax.imshow(img, cmap='RdGy', origin='lower', norm=SymLogNorm(0.5, 1, vmin=-pchi, vmax=pchi))
            ax.set_title(f'{filter} - PSF',fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
            # axes[0,0].set_ylabel(f'Object {objid}',fontsize=20)

    else:
        print('Filter not found.')



def get_scale_pkpc(objid, filter, redshift, catalog, filenames):
    """
    Return pkpc-per-pixel from RESOLUTION_PKPC in the primary header (HDU 0).
    Uses the same filename-selection logic as grab_image().
    """
    # find row in catalog
    idx = list(catalog['id']).index(objid)
    fileNumber = catalog['fileNumber'][idx]

    # choose correct file - SAME LOGIC AS grab_image()
    if redshift == 7:
        hsel = [(f'file{fileNumber}' in f) and (filter in f) and ('_nopsf' in f)
                for f in filenames]
    else:
        hsel = [(filter in f) and ('_nopsf' in f) for f in filenames]

    if not any(hsel):
        print(f"No matching file for ID={objid}, filter={filter}")
        return None

    fn = np.array(filenames)[hsel][0]

    # read primary header (HDU 0)
    with fits.open(fn) as hdul:
        hdr = hdul[0].header
        return hdr['RESOLUTION_PKPC']
    


def get_scale_as(objid, filter, redshift, catalog, filenames):
    """
    Return pkpc-per-pixel from RESOLUTION_PKPC in the primary header (HDU 0).
    Uses the same filename-selection logic as grab_image().
    """
    # find row in catalog
    idx = list(catalog['id']).index(objid)
    fileNumber = catalog['fileNumber'][idx]

    # choose correct file - SAME LOGIC AS grab_image()
    if redshift == 7:
        hsel = [(f'file{fileNumber}' in f) and (filter in f) and ('_nopsf' in f)
                for f in filenames]
    else:
        hsel = [(filter in f) and ('_nopsf' in f) for f in filenames]

    if not any(hsel):
        print(f"No matching file for ID={objid}, filter={filter}")
        return None

    fn = np.array(filenames)[hsel][0]

    # read primary header (HDU 0)
    with fits.open(fn) as hdul:
        hdr = hdul[0].header
        return hdr['RESOLUTION_ARCSEC']
