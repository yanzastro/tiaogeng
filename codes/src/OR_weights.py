import fitsio
from astropy.io import fits
import numpy as np
import matplotlib as mpl
import healpy as hp
import matplotlib.pyplot as plt
from pathos.pools import ProcessPool
from tqdm import tqdm
import sklearn.cluster
from scipy.spatial.distance import cdist
import os

os.environ['OMP_NUM_THREADS'] = '80'
os.environ['OPENBLAS_NUM_THREADS'] = '80'
os.environ['MKL_NUM_THREADS'] = '80'

class som2stats:
    def __init__(self, som):
        '''
        This class calculates necessary statistics by summarizing a pre-trained SOM.
        Input:
        som: a pre-trained somoclu som object;
        data: a N x M array, where N is the number of sources and M is the number of systematics;
        radec: RA and Dec of testing sources in degree.
        '''
        self.som = som
   
    def train_som(self, training_data, radiusN=5):
        self.training_data = training_data
        self.som.train(training_data, radiusN=radiusN)
        self.training_bmus = self.som.bmus
        self.som_dim = self.som.codebook.shape[0]
    
    def get_surface_state(self, data):
        """Return the Euclidean distance between codebook and data.
        :param data: Optional parameter to specify data, otherwise the
                     data used previously to train the SOM is used.
        :type data: 2D numpy.array of float32.
        :returns: The the dot product of the codebook and the data.
        :rtype: 2D numpy.array
        """

        d = data
        som=self.som
        codebookReshaped = som.codebook.reshape(
            som.codebook.shape[0] * som.codebook.shape[1], som.codebook.shape[2])
        parts = np.array_split(d, 200, axis=0)
        #am = np.empty((0, (som._n_columns * som._n_rows)), dtype="float64")
        am = np.zeros((data.shape[0], som._n_columns * som._n_rows))

        i = 0
        for part in tqdm(parts):
            am[i:i+part.shape[0]] = cdist((part), codebookReshaped, 'euclidean')
            i = i+part.shape[0]
        return am
    
    def get_OR_info(self, radec, testing_data=None, step=1000):
        '''
        This function gets the information needed for organized random from a pre-trained SOM. The main
        part is to calculate 'bmus' from a set of testing data
        It works by multiprocessing chunks of the data.
        Input:
        radec: an array of RA and Dec of the testing data set in degrees.
        testing_data: np.ndarray of the data vector. If not given, then use training data;
        step: int, the size of a chunk of the data.
        Output (attributes added):
        self.bmus: bmus correpsonding to the testing data
        '''
        self.radec = radec
        if testing_data is None:
            self.data = self.training_data
            self.bmus = self.training_bmus
            self.source_somcell_ind = self.som_ind_to_1d(self.bmus.T[0], self.bmus.T[1])
            return
        self.data = testing_data
        data = self.data
        som = self.som
        dmap = self.get_surface_state(data)
        self.bmus = som.get_bmus(dmap)

        self.source_somcell_ind = self.som_ind_to_1d(self.bmus.T[0], self.bmus.T[1])
        return 

    def get_test_hp_Nmap(self, Ns):     
        '''
        This function returns the healpix indices of sources in the testing data given Nside = Ns and the number count map.
        Attributes added:
        source_hp_ind: the healpix index of each source;
        source_hp_ind_unique: the unique healpix pixel indices.This array tells you which pixel contains at least one galaxy.
        Nmap: the number count map corresponding to the catalog.
        '''
        self.source_hp_ind = hp.ang2pix(Ns, self.radec.T[0], self.radec.T[1], lonlat=True)
        self.source_hp_ind_unique, N_p = np.unique(self.source_hp_ind, return_counts=True)
        # this line gives the unique healpix pixel indices that contains at least one galaxy, as well as the number of galaxies in each pixel 
        Nmap = np.zeros(hp.nside2npix(Ns))
        Nmap[self.source_hp_ind_unique] = N_p
        return Nmap
        
    def get_sys_maps(self):
        '''
        This function calculates the average value of each systematics in each SOM cell.
        It will add an DxDxM (D is the dimension of the SOM, M is the number of systematics) 
        array 'sys_maps' as a class attribute
        '''
        sys_maps = np.zeros((self.som_dim, self.som_dim, self.data.shape[1]))
        n_maps = np.zeros((self.som_dim, self.som_dim))
        
        try:
            winner_x, winner_y = self.bmus.T
        except AttributeError:
            print('bmus not calculated yet. now calculating...')
            self.get_test_bmus()
            winner_x, winner_y = self.bmus.T
            
        for i in range(len(self.data)):
            sys_maps[winner_x[i], winner_y[i]] += self.data[i]
            n_maps[winner_x[i], winner_y[i]] += 1
    
        sys_maps /= n_maps[:,:,None]
        self.sys_maps = sys_maps
    
    def hierarchical_clustering(self, n_cluster=100, algorithm=sklearn.cluster.AgglomerativeClustering):
        '''
        This function calculates the hierarchical clustering of the pre-trained SOM.
        Input:
        n_cluster: the number of clusters;
        algorithm: a clustering method from sklearn.cluster
        Output:
        self.som_cluster_ind: an 1-D array of cluster indices of each SOM cell.
        '''
        algorithm_ = algorithm(n_clusters=n_cluster, linkage='average')
        self.som.cluster(algorithm_)
        self.n_cluster = n_cluster
        self.som_cluster_ind = self.som.clusters.reshape(-1)
        
        try:
            winner_ind1d = self.source_somcell_ind
        except AttributeError:
            print('bmus not calculated yet. now calculating...')
            self.get_test_bmus()
            winner_ind1d = self.source_somcell_ind
            
        source_cluster_ind = np.zeros_like(winner_ind1d)
        for i in range(winner_ind1d.size):
            source_cluster_ind[i] = self.som_cluster_ind[winner_ind1d[i]]
        self.source_cluster_ind = source_cluster_ind
        
    def som_ind_to_1d(self, xi, yi):
        '''
        This function converts the 2-D indices of a SOM into a 1-D index of the flattend
        SOM. 
        Input:
        xi, yi: integers of the indices of a SOm
        '''
        return xi * self.som_dim + yi   
    
    
    def calculate_or_weights(self, Ns, pixfrac=None, selection=None):

        '''
        This function calculates the organized random weight on the pixelized sky.
        Input:
        Ns: an integer specifying the Nside of the weight map.
        pixfrac: a Healpix map specifying the pixel coverage in the footprint. It will be up/downgraded to Ns if it's Nside does not match Ns.
        selection: an 1-D array containing source indices that are used to select a subsample of sources to generate the OR weight.
        Output:
        weight_map: a healpix map of the organized random weight
        number_density: a 2-D matrix of number_density in each cell of the SOM
        '''
        som_dim = self.som_dim        
        Nmap = self.get_test_hp_Nmap(Ns, selection)
        
        if selection is None:
            select_ind = np.arange(self.radec.T[0].size)
        else:
            select_ind = selection
        source_hp_ind = self.source_hp_ind#[select_ind]

        try:
            som_cluster_ind1d = self.som_cluster_ind
        except AttributeError:
            print('hierarchical clustering not calculated yet. now calculating...')
            self.hierarchical_clustering()
            som_cluster_ind1d = self.som_cluster_ind

        try:
            source_cluster_ind = self.source_cluster_ind
        except AttributeError:
            print('source catalog has not been assigned to hierarchical clusters. now calculating...')
            self.get_test_cluster_ind()
            source_cluster_ind = self.source_cluster_ind
            
        if pixfrac is None:
            print("Warning: no input fraction file.")
            frac = np.zeros(hp.nside2npix(Ns))
            frac[self.source_hp_ind] = 1
        else:
            if pixfrac.size != hp.nside2npix(Ns):
                print("Nside of pixel fraction map does not match that of the OR weight map. Re-pixelizing it to be the same as the OR weight map.")
                pixfrac = hp.ud_grade(pixfrac, Ns)
            frac = pixfrac
        
        source_hp_ind_unique, N_p = np.unique(source_hp_ind, return_counts=True)
        # this line gives the unique healpix pixel indices that contains at least one galaxy, as well as the number of galaxies in each pixel 
        wmap = np.zeros(hp.nside2npix(Ns))
        #Nmap = self.Nmap
        source_hp_ind_unique = self.source_hp_ind_unique
        
        source_cluster_ind = self.source_cluster_ind[select_ind]
        
        A_pix = hp.nside2pixarea(Ns) * frac # the area of a Healpix pixel
        number_density = np.zeros(som_dim**2)
        for cluster_ind in tqdm(range(self.n_cluster)):
            source_clusteri_ind = np.where(source_cluster_ind==cluster_ind)[0]
            # pick out the catalog indices of sources that are in the cluster_ind'th hierarchical cluster
            N_i = source_clusteri_ind.size  # and the number of sources in that cluster        
            source_hp_ind_i = source_hp_ind[source_clusteri_ind]  # and the Healpix pixel indices
            source_hp_ind_clusteri_unique, N_p_i = np.unique(source_hp_ind_i, return_counts=True)
                                    
            f_p_i = N_p_i / Nmap[source_hp_ind_clusteri_unique]
            A_i = np.sum(f_p_i*A_pix[source_hp_ind_clusteri_unique])            
                
            n_i = N_i / A_i
            number_density[som_cluster_ind1d==cluster_ind] = n_i
            
            wmap[source_hp_ind_clusteri_unique] += n_i * A_pix[source_hp_ind_clusteri_unique] * f_p_i
            
        number_density = number_density.reshape(som_dim, som_dim)
        
        area = np.sum(frac)*hp.nside2pixarea(hp.npix2nside(frac.size), degrees=True)
        print(f'Footprint area is: {area} degree^2.')
        
        area_occ = np.sum(frac*(Nmap>0))*hp.nside2pixarea(hp.npix2nside(frac.size), degrees=True)
        
        frac_occupied = area_occ / area
        print('Fraction of occupied pixels: '+str(frac_occupied))
        return wmap, number_density
            
    
    def get_cluster_map(self, cluster_ind, pixfrac=None):
        som_dim = self.som_dim
        
        Nmap = self.get_test_hp_Nmap(Ns)
        source_hp_ind = self.source_hp_ind

        try:
            som_cluster_ind1d = self.som_cluster_ind
        except AttributeError:
            print('hierarchical clustering not calculated yet. now calculating...')
            self.hierarchical_clustering()
            som_cluster_ind1d = self.som_cluster_ind

        try:
            source_cluster_ind = self.source_cluster_ind
        except AttributeError:
            print('source catalog has not been assigned to hierarchical clusters. now calculating...')
            self.get_test_cluster_ind()
            source_cluster_ind = self.source_cluster_ind
            
        if pixfrac is None:
            print("Warning: no input fraction file.")
            frac = np.zeros(hp.nside2npix(Ns))
            frac[self.source_hp_ind] = 1
        else:
            if pixfrac.size != hp.nside2npix(Ns):
                print("Nside of pixel fraction map does not match that of the OR weight map. Re-pixelizing it to be the same as the OR weight map.")
                pixfrac = hp.ud_grade(pixfrac, Ns)
            frac = pixfrac
        
        source_hp_ind_unique, N_p = np.unique(source_hp_ind, return_counts=True)
        # this line gives the unique healpix pixel indices that contains at least one galaxy, as well as the number of galaxies in each pixel 
        area_map = np.zeros(hp.nside2npix(Ns))
        #Nmap = self.Nmap
        source_hp_ind_unique = self.source_hp_ind_unique
        A_pix = hp.nside2pixarea(Ns)# the area of a Healpix pixel
        number_contrast = np.zeros(som_dim**2)
        source_clusteri_ind = np.where(self.source_cluster_ind==cluster_ind)[0]
        # pick out the catalog indices of sources that are in the cluster_ind'th hierarchical cluster
        N_i = source_clusteri_ind.size  # and the number of sources in that cluster        
        source_hp_ind_i = self.source_hp_ind[source_clusteri_ind]  # and the Healpix pixel indices
        source_hp_ind_clusteri_unique, N_p_i = np.unique(source_hp_ind_i, return_counts=True)
        A_p_i = N_p_i / Nmap[source_hp_ind_clusteri_unique] * A_pix
        area_map[source_hp_ind_clusteri_unique] = A_p_i 
        
        number_contrast = number_contrast.reshape(som_dim, som_dim)
        frac_occupied = (Nmap>0).sum() / (frac>0).sum()
        print('Fraction of occupied pixels: '+str(frac_occupied))
        return area_map
    
    def calculate_ngal_i(self, Ns, pixfrac=None):

        '''
        This function calculates the galaxy number density for cluster i.
        Input:
        Ns: an integer specifying the Nside used to pixelize the sky.
        Output:
        ngal_i: an array containing galaxy number density for each cluster
        '''
        som_dim = self.som_dim

        Nmap = self.get_test_hp_Nmap(Ns)
        source_hp_ind = self.source_hp_ind

        try:
            som_cluster_ind1d = self.som_cluster_ind
        except AttributeError:
            print('hierarchical clustering not calculated yet. now calculating...')
            self.hierarchical_clustering()
            som_cluster_ind1d = self.som_cluster_ind

        try:
            source_cluster_ind = self.source_cluster_ind
        except AttributeError:
            self.get_test_cluster_ind()
            source_cluster_ind = self.source_cluster_ind
            
        if pixfrac is None:
            frac = np.zeros(hp.nside2npix(Ns))
            frac[self.source_hp_ind] = 1
        else:
            frac = pixfrac
        
        ngal_i = np.zeros(self.n_cluster)
        source_hp_ind_unique, N_p = np.unique(source_hp_ind, return_counts=True)
        # this line gives the unique healpix pixel indices that contains at least one galaxy, as well as the number of galaxies in each pixel 
        source_hp_ind_unique = self.source_hp_ind_unique
        A_pix = hp.nside2pixarea(Ns, degrees=True)*3600# the area of a Healpix pixel
        number_contrast = np.zeros(som_dim**2)
        for cluster_ind in range(self.n_cluster):
            source_clusteri_ind = np.where(self.source_cluster_ind==cluster_ind)[0]
            # pick out the catalog indices of sources that are in the cluster_ind'th hierarchical cluster
            N_i = source_clusteri_ind.size  # and the number of sources in that cluster        
            source_hp_ind_i = self.source_hp_ind[source_clusteri_ind]  # and the Healpix pixel indices
            source_hp_ind_clusteri_unique, N_p_i = np.unique(source_hp_ind_i, return_counts=True)
            A_p_i = N_p_i / Nmap[source_hp_ind_clusteri_unique]
            A_i = np.sum(N_p_i / Nmap[source_hp_ind_clusteri_unique]*frac[source_hp_ind_clusteri_unique]*A_pix)
            ngal_i[cluster_ind] = N_i / A_i
        ngal_tot = N_p.sum() / np.sum(A_pix * frac[source_hp_ind_unique])
        return ngal_i, ngal_tot

    def calculate_median_sys(self):

        '''
        This function calculates the median systematic value for each cluster.
        Input:
        Ns: an integer specifying the Nside used to pixelize the sky.
        New attributes:
        ngal_i: an array containing galaxy number density for each cluster
        '''
        som_dim = self.som_dim

        try:
            source_hp_ind = self.source_hp_ind
        except AttributeError:
            Nmap = self.get_test_hp_Nmap(Ns)
            source_hp_ind = self.source_hp_ind

        try:
            som_cluster_ind1d = self.som_cluster_ind
        except AttributeError:
            print('hierarchical clustering not calculated yet. now calculating...')
            self.hierarchical_clustering()
            som_cluster_ind1d = self.som_cluster_ind

        try:
            source_cluster_ind = self.source_cluster_ind
        except AttributeError:
            self.get_test_cluster_ind()
            source_cluster_ind = self.source_cluster_ind
        
        self.ngal_i = np.zeros(self.n_cluster)
        source_hp_ind_unique, N_p = np.unique(source_hp_ind, return_counts=True)
        # this line gives the unique healpix pixel indices that contains at least one galaxy, as well as the number of galaxies in each pixel 
        sysmed_i = np.zeros((self.n_cluster, self.data.shape[1]))
        source_hp_ind_unique = self.source_hp_ind_unique
        for cluster_ind in range(self.n_cluster):
            sysmed_i[cluster_ind] = np.median(self.data[self.source_cluster_ind==cluster_ind], axis=0)
        return sysmed_i
