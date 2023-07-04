# This file contains functions to generate mock catalogs from GLASS as well as pixelizing a catalog.

import treecorr
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
# these are the GLASS imports: cosmology and everything in the glass namespace
from cosmology import Cosmology
from astropy.io import fits
import healpy as hp
# also needs camb itself to get the parameter object
import camb
import sys
import glass.shells
import glass.fields
import glass.points
import glass.shapes
import glass.lensing
import glass.galaxies
import glass.observations
import glass.ext.camb
from tqdm import tqdm


def glass_mock(Oc, Ob, h, bias, z, n_arcmin2, dndz, vis, nside, outfile, gls=None, random=False):
    '''
    This function using GLASS to generate a mock galaxy catalog under a given cosmology and redshift distribution.
    Note that the version of glass for this function is the version "2023.06".
    Input:
    Oc, Ob, h, bias: cosmological parameters;
    z, dndz: redshift distribution of the galaxy catalog (needs to be normalized);
    n_arcmin2: mean galaxy surface number density in ngal/arcmin^2;
    vis: a binary sky mask in Healpix format that defines the footprint of the survey;
    outfile: a string of the path to save the catalog;
    random: bool, whether to generate a galaxy catalog or a uniform random catalog.
    
    '''

    dz = (z[-1]-z[0])/z.size
    dndz /= (dndz.sum()*dz)
    dndz *= n_arcmin2    
    
    # shells of 200 Mpc in comoving distance spacing
    zb = glass.shells.redshift_grid(z[0], z[-1], dz=0.1)

    # tophat window functions for shells
    ws = glass.shells.tophat_windows(zb)
    
    # compute the angular matter power spectra of the shells with CAMB
    if gls is None:
        lmax = 3 * nside
        pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, WantTransfer = True)
        cls = glass.ext.camb.matter_cls(pars, lmax, ws)

        # compute Gaussian cls for lognormal fields for 3 correlated shells
        # putting nside here means that the HEALPix pixel window function is applied
        gls = glass.fields.lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=3)

    # generator for lognormal matter fields
    matter = glass.fields.generate_lognormal(gls, nside)    
    
    lon = np.array([])
    lat = np.array([])
    Z = np.array([])
    # simulate the matter fields in the main loop, and build up the catalogue
    for i, delta_i in tqdm(enumerate(matter), total=zb.size-1):

        # the true galaxy distribution in this shell
        z_i, dndz_i = glass.shells.restrict(z, dndz, ws[i])

        # integrate dndz to get the total galaxy density in this shell
        ngal = np.trapz(dndz_i, z_i)
        if random==True:
            delta_i = np.zeros_like(delta_i)
    
        # generate galaxy positions from the matter density contrast
        for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(ngal, delta_i, bias, vis):

            # generate random redshifts from the provided nz
            gal_z = glass.galaxies.redshifts_from_nz(gal_count, z_i, dndz_i)

            # make a mini-catalogue for the new rows
            
            lon = np.hstack([lon, gal_lon])
            lat = np.hstack([lat, gal_lat])
            Z = np.hstack([Z, gal_z])
            
    print(str(len(lon))+' galaxies are generated!')
  
    col_ra = fits.Column(name='RA', format='D', array=np.array(lon))
    col_dec = fits.Column(name='Dec', format='D', array=np.array(lat))
    col_redshift = fits.Column(name='Redshift', format='D', array=np.array(Z))
    coldefs = fits.ColDefs([col_ra, col_dec, col_redshift])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto(outfile, overwrite=True)

    return


def cat_to_hpcat(infile, outfile, Ns, racol, deccol, wcol=None, mask=None):
    '''
    This function reads sources from a catalog and pixelize them into Healpix grids, 
    and then save the RA and Dec of HP pixels that contain at least one source as well as the number of sources in those pixels.
    Input:
    infile, outfile: strings defining the input and output catalog files;
    Ns: Nside of the healpix grid;
    racol, deccol: the columns of RA and Dec in the input file;
    wcol, mask: the columns of weight and mask in the input file
    '''
    infile_ = fits.open(infile)
    
    lon = infile_[1].data[racol]
    lat = infile_[1].data[deccol]
    if wcol is not None:
        w = np.array(infile_[1].data[wcol])
    else:
        w = np.ones_like(lon)
    
    hp_ind = hp.ang2pix(Ns, lon, lat, lonlat=True) 
    gmap = np.zeros(hp.nside2npix(Ns))
    np.add.at(gmap, hp_ind, w)
    
    #gmap /= hp.nside2pixarea(2048)
    #n_bar = lon.size / np.pi / 4
    
    if mask is None:
        mask = np.ones_like(gmap)
    gmap *= mask
    hp_ind_unmasked = np.arange(mask.size)[(mask!=0)*(gmap>0)]
    hp_lon, hp_lat = hp.pix2ang(Ns, hp_ind_unmasked, lonlat=True)
    delta = gmap[hp_ind_unmasked]
    
    print(str((delta!=0).size)+' pixels contain galaxies!')

    col_ra = fits.Column(name='RA', format='D', array=hp_lon)
    col_dec = fits.Column(name='Dec', format='D', array=hp_lat)
    delta = fits.Column(name='Delta', format='D', array=delta)
    hp_ind = fits.Column(name='hp_ind', format='K', array=hp_ind_unmasked)
    coldefs = fits.ColDefs([col_ra, col_dec, delta, hp_ind])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto(outfile, overwrite=True)
    
    return

def cat_to_hpmap(infile, outfile, Ns, racol, deccol, wcol=None, mask=None):
    '''
    This function reads sources from a catalog and pixelize them into a Healpix map, 
    and then save the RA and Dec of HP pixels that contain at least one source as well as the number of sources in those pixels.
    Input:
    infile, outfile: strings defining the input and output catalog files;
    Ns: Nside of the healpix grid;
    racol, deccol: the columns of RA and Dec in the input file;
    wcol, mask: the columns of weight and mask in the input file
    Return:
    A healpix map
    '''
    infile_ = fits.open(infile)
    
    lon = infile_[1].data[racol]
    lat = infile_[1].data[deccol]
    if wcol is not None:
        w = np.array(infile_[1].data[wcol])
    else:
        w = np.ones_like(lon)
    
    hp_ind = hp.ang2pix(Ns, lon, lat, lonlat=True) 
    gmap = np.zeros(hp.nside2npix(Ns))
    np.add.at(gmap, hp_ind, w)
    
    #gmap /= hp.nside2pixarea(2048)
    #n_bar = lon.size / np.pi / 4
    
    if mask is None:
        mask = np.ones_like(gmap)
    gmap *= mask
    return gmap


def hpmap_to_hpcat(hpmap, outfile, Ns, mask=None):

    '''
    This function converts a Healpix map into a catalog in which RA and Dec are HP pixels that contain 
    at least one source as well as the number of sources in those pixels.
    hpmap: a healpix map;
    outfile: the path to the output catalog;
    Ns: the Nside of the input healpix map;
    mask: another healpix map defining the mask to be applied to the catalog
    '''
    gmap = hpmap    
    #gmap /= hp.nside2pixarea(2048)
    #n_bar = lon.size / np.pi / 4
    
    if mask is None:
        mask = np.ones_like(gmap)
    #n_bar = np.mean(gmap[mask!=0])
    #gmap = (gmap - n_bar)/n_bar
    gmap *= mask
    hp_ind_unmasked = np.arange(mask.size)[(mask!=0)*(gmap!=0)]
    hp_lon, hp_lat = hp.pix2ang(Ns, hp_ind_unmasked, lonlat=True)
    delta = gmap[hp_ind_unmasked]
    
    print(str((delta!=0).size)+' pixels contain galaxies!')

    col_ra = fits.Column(name='RA', format='D', array=hp_lon)
    col_dec = fits.Column(name='Dec', format='D', array=hp_lat)
    delta = fits.Column(name='Delta', format='D', array=delta)
    hp_ind = fits.Column(name='hp_ind', format='K', array=hp_ind_unmasked)
    coldefs = fits.ColDefs([col_ra, col_dec, delta, hp_ind])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto(outfile, overwrite=True)
    
    return

