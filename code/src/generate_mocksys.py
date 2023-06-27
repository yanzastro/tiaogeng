### This file contains functions and classes to generate tiles in mock galaxy catalogues as well as corresponding depletion functions

import numpy as np
import healpy as hp
from astropy.io import fits
from tqdm import tqdm
#from pathos.multiprocessing import Pool
import time 
import os
from scipy.interpolate import NearestNDInterpolator

np.random.seed(42)

def gaussian_2d(x, y, cov=np.eye(2)*5, xmean=0, ymean=0, amp=1, shift=0):
    '''
    This function returns the 2-D Gaussian-like function of x and y. One might specify the covariance matrix, x and y mean values, amplitudes and shift.
    Input:
    x, y: arrays with same length;
    cov: a 2*2 positive definite, symmetric matrix, default set to be unit matrix;
    xmean, ymean: floats, the mean value for x and y, default values are zero;
    amp: the amplitude of the Gaussian. default value set to be one;
    shift: the overall shift of the function. default value set to be zero.
    Output:
    an array with the same length as x and y.
    '''
    dx = x - xmean
    dy = y - ymean
    dxdy = np.vstack([dx, dy])
    chi2 = np.sum(dxdy * (cov @ dxdy), axis=0)
    return np.exp(-chi2/2) * amp + shift


class trig_func:
    def __init__(self, A, T, phi, b):
        '''
        Trigonometric functions.
        A: amplitude;
        T: period;
        phi: initial phasel
        b: shift
        Returns A*sin(2*pi/T*x + phi) + b
        '''
        self.A = A
        self.w = 2*np.pi/T
        self.phi = phi
        self.b = b
    def __call__(self, x):
        return self.A * np.sin(self.w*x+self.phi) + self.b


def downsample(n_input, n_output):
    '''
    This function downsamples a sample into desired output number.
    Inputs:
    n_input: int, the size of input sample;
    n_output: int, the size of output sample;
    Outputs:
    an array with size n_input with 0 for removing a source and 1 for keeping a source
    
    '''
    if n_output > n_input:
        raise ValueError("n_output should be less than n_input!")
    frac = n_output / n_input * 1.0
    rand = np.random.random(size=n_input)
    keep = (rand<frac)
    return keep

def lon_diff(lon1, lon2):
    '''
    Calculate the minor arc difference between two lontitudes.
    '''
    #lon1 = np.atleast_1d(lon1)
    #lon2 = np.atleast_1d(lon2)
    dlon = lon1 - lon2
    
    dlon[(dlon)>180] -= 360
    dlon[(dlon)<-180] += 360
    
    return dlon

def lon_sum(lon, dlon):
    '''
    Calculate the longitude by adding a longitude difference to another longutude.
    dlon has to be within -180deg and 180 deg.
    '''
    lon = np.atleast_1d(lon)
    dlon = np.atleast_1d(dlon)
    aux_sign = np.zeros_like(lon + dlon)
    aux_sign[(lon+dlon<360)*(lon+dlon>0)] = 0
    aux_sign[lon+dlon>360] =-1
    aux_sign[lon+dlon<0] = 1
    return lon + dlon + 360 * aux_sign

class tiles:
    def __init__(self, start_lon, start_lat, dx, dy, nlon, nlat):
        '''
        A class that defines a group of tiles in the sky.
        Inputs:
        start_lon, start_lat: the coordinate of the starting point in the sky in degrees;
        dx, dy: the size of the x and y axis of the tile (approximately the great circle arc) in degrees;
        nlon, nlat: number of tiles in longitude and latitide.
        Attributes:
        n_tiles: the total number of tiles (equals to nlon * nlat);
        tile_centers: a n_tiles * 2 array containing the coordinates of centers of all the tiles in degrees;
        corner_lon_w, corner_lon_e: the longitudes of the western and eastern corners of each tile;
        corner_lat_n, corner_lat_s: the latitudes of the northern and southern corners of each tile;
        '''
        self.n_tiles = nlon * nlat
        lonind = np.arange(nlon)
        latind = np.arange(nlat)
        lonind, latind = np.meshgrid(lonind, latind)

        lonind = lonind.reshape(-1)
        latind = latind.reshape(-1)
        center_lats = start_lat + latind * dy
        center_lons = lon_sum(start_lon, dx * lonind / np.cos(np.radians(center_lats)))
        self.tile_centers = np.vstack([center_lons, center_lats]).T
        self.corner_lon_w = lon_diff(self.tile_centers.T[0], dx/np.cos(np.radians(self.tile_centers.T[1]))/2)
        self.corner_lon_e = lon_sum(self.tile_centers.T[0], dx/np.cos(np.radians(self.tile_centers.T[1]))/2)
        self.corner_lat_n = self.tile_centers.T[1]+dy/2
        self.corner_lat_s = self.tile_centers.T[1]-dy/2
        self.dlats = self.corner_lat_n - self.corner_lat_s
        self.dlons = lon_diff(self.corner_lon_e, self.corner_lon_w)
        
    def get_tileind(self, lon, lat):
        '''
        This function returns the tile index of a given source at (lon, lat).
        Inputs:
        lon, lat: coordinate of the source in degrees.
        Output:
        The tile index that the source belongs to. If it does not belong to any of the tiles, return -1.
        '''
        
        tile_inds = np.ones_like(lon) * (-1.)
        source_inds = np.arange(lon.size)
        
        ind_unasigned = np.where(tile_inds==-1.)[0]
        
        for i in tqdm(range(self.corner_lon_w.size)): 
                        
            lon_unasigned = lon[ind_unasigned]
            lat_unasigned = lat[ind_unasigned]
            
            dlon = lon_unasigned - self.tile_centers[i][0]
            dlon[(dlon)>180] -= 360
            dlon[(dlon)<-180] += 360
            diff_to_center_X = np.abs(dlon)
                        
            diff_to_center_Y = np.abs(lat_unasigned - self.tile_centers[i][1])
                        
            intile = (diff_to_center_X < self.dlons[i] / 2) * (diff_to_center_Y < self.dlats[i] / 2)

            intile_i = ind_unasigned[intile]
            tile_inds[intile_i] = i
            ind_unasigned = np.where(tile_inds==-1.)[0]
                       
        return tile_inds

        
    def get_tilecoord(self, lon, lat):
        tileind = self.get_tileind(lon, lat)
        if tileind == -1:
            print("Warning: The source is not in any of the tile!")
            return 360, 360
        else:
            return lon_diff(lon, self.tile_centers[tileind][0]) * np.cos(np.radians(self.tile_centers[tileind][1])), lat - self.tile_centers[tileind][1]
        
    def save_tilefits(self, outfile):
        col_centers_lon = fits.Column(name='centers_lon', format='E', array=self.tile_centers.T[0])
        col_centers_lat = fits.Column(name='centers_lat', format='E', array=self.tile_centers.T[1])
        
        col_w_lon = fits.Column(name='west_lon', format='E', array=self.corner_lon_w)
        col_e_lon = fits.Column(name='east_lon', format='E', array=self.corner_lon_e)
        col_n_lat = fits.Column(name='north_lat', format='E', array=self.corner_lat_n)
        col_s_lat = fits.Column(name='south_lat', format='E', array=self.corner_lat_s)
        
        coldefs = fits.ColDefs([col_centers_lon, col_centers_lat, col_w_lon, col_e_lon, col_n_lat, col_s_lat])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.writeto(outfile, overwrite=True)

        
class tiles_fromcorner(tiles):
    '''
        A class that defines a group of tiles in the sky. It is a subclass of 'tiles'. The inputs are the four ndarrays specifying the four corners of the tiles.
        Inputs:
        start_lon, start_lat: the coordinate of the starting point in the sky in degrees;
        dx, dy: the size of the x and y axis of the tile (approximately the great circle arc) in degrees;
        nlon, nlat: number of tiles in longitude and latitide.
        Attributes:
        n_tiles: the total number of tiles (equals to nlon * nlat);
        tile_centers: a n_tiles * 2 array containing the coordinates of centers of all the tiles in degrees;
        corner_lon_w, corner_lon_e, corner_lat_n, corner_lat_s: ndarrays, the longitudes of the western and eastern corners of each tile and the latitudes of the northern and southern corners of each tile;
    '''
    def __init__(self, corner_lon_e, corner_lon_w, corner_lat_s, corner_lat_n):
        self.corner_lon_w = corner_lon_w
        self.corner_lon_e = corner_lon_e
        self.corner_lat_n = corner_lat_n
        self.corner_lat_s = corner_lat_s
        self.n_tiles = corner_lat_s.size
        self.dlats = corner_lat_n - corner_lat_s
        self.dlons = lon_diff(corner_lon_e, corner_lon_w) 

        center_lats = (corner_lat_n + corner_lat_s) / 2
        center_lons = lon_diff(corner_lon_e, self.dlons/2)
        self.tile_centers = np.vstack([center_lons, center_lats]).T
        

class sys_in_tile:
    """
    A class that defines systematic value and probability to keep a source accordingly.
    General input:
    tiles: a tiles object;
    config: a configuration dictionary for each type of systematics (see the following for specific instruction for each type of systematics);
    
    By calling this class, the systematics value and keeping probability with be returned.
    """
    def __init__(self, tiles, config):
        self.tiles = tiles
        self.config = config
    def eval_sys(self, source_lons, source_lats, source_tile_ids):
        return 
    
    def reject_or_keep(self, sys, rejfunc):
        '''
        A function that decides whether a source is rejected or kept according to the rejfunc.
        Input: 
        sys: an array of systematics for each source;
        Output:
        an array with 1 (for keepking the source) or 0 (for removing the source)
        rejfunc: a function of systematic value that returns the probability to keep the source with that value of systematics.

        '''
        sys = np.atleast_1d(sys)
        rand = np.random.random(size=sys.size)
        p = rejfunc(sys)
        keep = (rand>p)
        p[p<0] = 0
        p[p>1] = 1
        keep[np.isnan(sys)] = 0

        return p, keep
    
    def __call__(self, source_lons, source_lats, source_tile_ids, rejfunc, normed=False):
        
        source_sys = self.eval_sys(source_lons, source_lats, source_tile_ids)
        if normed:
            source_sys -= source_sys[~np.isnan(source_sys)].min()
            source_sys /= (source_sys[~np.isnan(source_sys)].max() - source_sys[~np.isnan(source_sys)].min())
        else: pass
        p, keep = self.reject_or_keep(source_sys, rejfunc)
        return source_sys, p, keep

    
class type_A_sys(sys_in_tile):
    '''
    This is the "type A" systematics, a large-scale distributed systematic which mimics the Galactic effect. We model it as a Gaussian centered at a point in the sky.
    Input config dictionary example:
    {'lon_sys_center': -5,  # the longitude of the systmatics center point 
                'lat_sys_center': 5,  # the latitude of the systmatics center point 
                'cov_xx': 0.0,  # the xx component of the covariance matrix of the Gaussian
                'cov_yy': .2,  # the yy component of the covariance matrix of the Gaussian
                'cov_xy': 0}  # the xy component of the covariance matrix of the Gaussian
    '''
    __doc__ = sys_in_tile.__doc__ + __doc__
    def __init__(self, tiles, config_sys_a):
        super(type_A_sys, self).__init__(tiles, config_sys_a)
            
    def eval_sys(self, source_lons, source_lats, source_tile_ids):
        x = lon_diff(source_lons, self.config['lon_sys_center'])
        y = source_lats - self.config['lat_sys_center']
        cov = np.array([[self.config['cov_xx'], self.config['cov_xy']],[self.config['cov_xy'], self.config['cov_yy']]])
        source_sys = gaussian_2d(x, y, cov)
        source_in_tilet = np.where(source_tile_ids==-1)[0]
        source_sys[source_in_tilet] = np.nan
        return source_sys
    
    
class type_B_sys(sys_in_tile):
    '''
    This is the "type B" systematics, varying as a 2-D Gaussian in each tile independently across tiles, mimicking telescope and camera effects such as PSF variations over the focal plane.
    Input config dictionary example:
    config_sys_b = {'xmean_fluc':.1,  # fluctuation of the center of the Gaussian across tiles
                'ymean_fluc':.1,  # fluctuation of the center of the Gaussian across tiles
                'covxx_mean': 10,  # mean and fluctuation in covariance terms
                'covyy_mean': 10,  # mean and fluctuation in covariance terms
                'covxx_fluc': 1,  # mean and fluctuation in covariance terms
                'covyy_fluc': 1,  # mean and fluctuation in covariance terms
                'covxy_fluc': 4,  # mean and fluctuation in covariance terms
                'amp': -1,  # the amplitude of the Gaussian
                'shift': 1}  # the shift of the Gaussian
    '''
    __doc__ = sys_in_tile.__doc__ + __doc__
    def __init__(self, tiles, config_sys_b):
        super(type_B_sys, self).__init__(tiles, config_sys_b)
        self.syscovs = np.zeros((tiles.n_tiles, 2, 2))
        self.xmeans = np.zeros((tiles.n_tiles))
        self.ymeans = np.zeros((tiles.n_tiles))
        for t in range(self.tiles.n_tiles):            
            cov_xx = self.config['covxx_mean'] + (np.random.rand() - 0.5) * 2 * self.config['covxx_fluc']
            cov_yy = self.config['covyy_mean'] + (np.random.rand() - 0.5) * 2 * self.config['covyy_fluc']
            cov_xy = (np.random.rand() - 0.5) * 2 * self.config['covxy_fluc']
            self.syscovs[t] = np.array([[cov_xx, cov_xy],[cov_xy, cov_yy]])
            self.xmeans[t] = np.random.normal(scale=self.config['xmean_fluc'])
            self.ymeans[t] = np.random.normal(scale=self.config['ymean_fluc'])
            
    def eval_sys(self, source_lons, source_lats, source_tile_ids):
        source_sys = np.zeros_like(source_tile_ids)
        for t in tqdm(range(self.tiles.n_tiles)):     
            cov = self.syscovs[t]
            xmean = self.xmeans[t]
            ymean = self.ymeans[t]
            source_in_tilet = np.where(source_tile_ids==t)[0]
            x = lon_diff(source_lons[source_in_tilet], self.tiles.tile_centers[t][0]) * np.cos(np.radians(self.tiles.tile_centers[t][1]))
            y = source_lats[source_in_tilet] - self.tiles.tile_centers[t][1]
            source_sys[source_in_tilet] = gaussian_2d(x, y, cov=cov, xmean=xmean, ymean=ymean, amp=self.config['amp'], shift=self.config['shift'])
        source_in_tilet = np.where(source_tile_ids==-1)[0]
        source_sys[source_in_tilet] = np.nan
        return source_sys
    
class type_C_sys(sys_in_tile):
    '''
    Type C systematics: varying randomly and independently between tiles while being uniform within each tile, mimicking per-exposure effects such as limiting depth variations that arise from the use of a step-and-stare observing strategy.
    configuration dictionary is not necessary.
    Input config dictionary example:
    config_sys_c = {'upper': 1  # the upper bound of random systematics;
    'lower': 0  # the lower bound of random systematics}
    '''
    __doc__ = sys_in_tile.__doc__ + __doc__
    def __init__(self, tiles, config_sys_c=None):
        super(type_C_sys, self).__init__(tiles, config_sys_c)
        if config_sys_c is None:
            self.system_intile = np.random.random(size = self.tiles.n_tiles)
        else:
            self.system_intile = np.random.random(size = self.tiles.n_tiles) * (self.config['upper'] - self.config['lower']) + self.config['lower']
    def eval_sys(self, source_lons, source_lats, source_tile_ids):
        source_sys = np.zeros(source_tile_ids.size)
        for t in range(self.tiles.n_tiles):
            source_in_tilet = np.where(source_tile_ids==t)[0]
            source_sys[source_in_tilet] = self.system_intile[t]
        source_in_tilet = np.where(source_tile_ids==-1)[0]
        source_sys[source_in_tilet] = np.nan  
        return source_sys
    

class real_to_mock:
    '''
    This class creates mock systematics for a mock galaxy catalog by doing the nearest-neighbour interpolation from a real catalog. It is needed for the 'data driven systematics test'.
    Input:
    ra, dec: ndarrays, RA and Dec of the real galaxy catalog;
    sys_values: ndarray, values of systematics of each galaxy;
    sys_names: names of these systematics  ### needed to be updated to a dictionary.
    '''
    def __init__(self, ra, dec, sys_values, sys_names):
        self.ra = ra
        self.dec = dec
        self.sys_names = sys_names
        self.sys_values = sys_values
        self.vec = hp.ang2vec(ra, dec, lonlat=True)
            
    def get_nn_interp(self):
        '''
        Construct nearst-neighbour interpolation object for each systematic.
        '''
        self.interp_func = {}        
        for i, sys_name in enumerate(self.sys_names):
            print(sys_name)
            self.interp_func[sys_name] = NearestNDInterpolator(self.vec, self.sys_values[i])
    
    def get_interpd_sys(self, ra_mock, dec_mock):
        '''
        Calculate the interpolated systematics values for a mock catalog.
        Input:
        ra_mock, dec_mock: ndarrays, the RA and Dec of the mock sample.
        '''
        mock_sys = {}
        vec = hp.ang2vec(ra_mock, dec_mock, lonlat=True)
        vec_inds = np.arange(ra_mock.size)
        
        for i, sys_name in enumerate(self.sys_names):
            print(sys_name)
            mock_sys[sys_name] = self.interp_func[sys_name](vec.T[0], vec.T[1], vec.T[2])
        return mock_sys
    
    @staticmethod
    def get_sys_cols(mock_sys):
        '''
        A static method to construct a list of fits table columns for an input dictionary of systematics
        '''
        col_list = []
        for sys_name in mock_sys.keys():
            col_list.append(fits.Column(name=sys_name, format='D', array=mock_sys[sys_name]))
        return col_list
    
    @staticmethod
    def get_keep_prob(ra, dec, delta_map, out_number=None, m=1):
        '''
        A static method to construct a list of fits table columns for an input dictionary of systematics
        Input:
            ra, dec: RA and Dec of input sources in degrees;
            delta_map: a Healpix map specifies the number contrast given by the selection map;
            out_number: None or an integer (preferantially smaller than the maximum of delta_map+1) that gives the number of post-selection sources that one wants to keep.
                if None, then return the keeping probability given only by delta_map.
            m: a float that intensifies (when m>0) or softifies (when m<1) the selection
        '''
        in_number = ra.size
        keep_frac = out_number / (in_number * 1.0)
        ns = hp.npix2nside(delta_map.size)
        hp_idx = hp.ang2pix(ns, ra, dec, lonlat=True)
        delta = delta_map[hp_idx]
        keep_out = keep_frac * (1+m*delta)
        if keep_out.max()>1:
            print("Warning: the maximum keeping probability is greater than 1!!!")
        elif keep_out.min()<0:
            print("Warning: the minimum keeping probability is lower than 0!!!")
        return keep_out
