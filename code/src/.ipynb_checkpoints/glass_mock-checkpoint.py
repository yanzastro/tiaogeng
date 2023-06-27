# This file contains functions to generate mock catalogs from GLASS as well as pixelizing a catalog.

import treecorr
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
# these are the GLASS imports: cosmology and everything in the glass namespace
from cosmology import Cosmology
import glass.all
import glass
from astropy.io import fits
import healpy as hp
# also needs camb itself to get the parameter object
import camb
import sys


def glass_mock(Oc, Ob, h, z, n_arcmin2, dndz, vis, nside, outfile, random=False, glass_output=None):
    '''
    This function using GLASS to generate a mock galaxy catalog under a given cosmology and redshift distribution.
    Note that the version of glass for this function is the commit '95fb2bbe27c1597b20b0c9c4b86b7f2dd1237cc9' of glass-dev/glass
    The associated 'gaussiancl' should be release 2022.7.28.
    Input:
    Oc, Ob, h: cosmological parameters;
    z, dndz: redshift distribution of the galaxy catalog (needs to be normalized);
    n_arcmin2: mean galaxy surface number density in ngal/arcmin^2;
    vis: a binary sky mask in Healpix format that defines the footprint of the survey;
    outfile: a string of the path to save the catalog;
    random: bool, whether to generate a galaxy catalog or a uniform random catalog.
    
    '''
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, WantTransfer = True)
    cosmo = Cosmology.from_camb(pars)

    lmax = 3 * nside
    
    lon = np.array([])
    lat = np.array([])
    Z = np.array([])
    if random is False:
        if glass_output is not None:
            generators = [
            glass.core.load(glass_output),
            glass.observations.vis_constant(vis, nside=nside),
            glass.galaxies.gal_b_const(1),
            glass.matter.lognormal_matter(nside),
            glass.galaxies.gal_density_dndz(z, n_arcmin2*dndz),
            glass.galaxies.gal_positions_mat(),
            glass.galaxies.gal_redshifts_nz(),        
            ]
        else:
            generators = [
            glass.cosmology.zspace(z[0], z[-1], dz=0.1),
            glass.matter.mat_wht_density(cosmo),
            glass.camb.camb_matter_cl(pars, lmax),
            glass.observations.vis_constant(vis, nside=nside),
            glass.galaxies.gal_b_const(1),
            glass.matter.lognormal_matter(nside),
            glass.galaxies.gal_density_dndz(z, n_arcmin2*dndz),
            glass.galaxies.gal_positions_mat(),
            glass.galaxies.gal_redshifts_nz(),        
            ]
        # simulate and add galaxies in each matter shell to cube
        for shell in glass.core.generate(generators):
            lon = np.hstack([lon, shell[glass.galaxies.GAL_LON]])
            lat = np.hstack([lat, shell[glass.galaxies.GAL_LAT]])
            Z = np.hstack([Z, shell[glass.galaxies.GAL_Z]])
        
    else:
        generators = [        
        glass.cosmology.zspace(z[0], z[-1], dz=0.1),
        glass.galaxies.gal_density_dndz(z, n_arcmin2*dndz),
        glass.galaxies.gal_positions_unif(),
        glass.galaxies.gal_redshifts_nz(),        
        ]        
    
        # simulate and add galaxies in each matter shell to cube
        
        for shell in glass.core.generate(generators):
            lon = np.hstack([lon, shell[glass.galaxies.GAL_LON]])
            lat = np.hstack([lat, shell[glass.galaxies.GAL_LAT]])
            Z = np.hstack([Z, shell[glass.galaxies.GAL_Z]])
            
        hp_ind = hp.ang2pix(nside, lon, lat, lonlat=True) 
        lon = lon[vis[hp_ind]>0]
        lat = lat[vis[hp_ind]>0]
        Z = Z[vis[hp_ind]>0]
        
    print(str(lon.size)+' galaxies are generated!')
  
    col_ra = fits.Column(name='RA', format='D', array=lon)
    col_dec = fits.Column(name='Dec', format='D', array=lat)
    col_redshift = fits.Column(name='Redshift', format='D', array=Z)
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

