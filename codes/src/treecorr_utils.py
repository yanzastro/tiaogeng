import numpy as np
import treecorr
import gc

def treecorr_NNcor(catd, catr, min_sep=3, max_sep=300, nbins=20, bin_slop=0.0, sep_units='arcmin', var_method='shot'):
    '''
    This function is a wrapper calling treecorr.NNCorrelation to calculate the galaxy correlation function as well as its covariance matrix.
    Input:
    catd, catr: treecorr Catalog objects that defines the data catalog and random catalog;
    other inputs are those for the NNCorrelation function.
    '''
    gg = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units=sep_units, var_method=var_method)
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units=sep_units, var_method=var_method)
    dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units=sep_units, var_method=var_method)
    rd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, bin_slop=bin_slop, sep_units=sep_units, var_method=var_method)
    gg.process(catd)
    rr.process(catr)
    dr.process(catd, catr)
    rd.process(catr, catd)
    w, w_err = gg.calculateXi(rr=rr, dr=dr, rd=rd)
    cov = gg.cov
    theta = gg.meanr
    del gg, rr, dr, rd
    gc.collect()
    return theta, w, cov

def mask_arrays(array, mask):
    '''
    This function removes terms in an array specified by 'mask'.
    Input:
    array: an ndarray to be masked. It has to have same size on each axis.
    mask: an array with bool terms that has the same length as any axis in 'array'. True means to remove.
    '''
    dim = array.ndim
    array_masked = array.copy()
    for i in range(dim):
        array_masked = np.delete(array_masked, mask, axis=i)
    return array_masked