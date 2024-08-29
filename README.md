# tiaogeng: recovering organized randoms for galaxy clustering measurements

## Introduction

This is a python package to calculate the ''organized random'' for galaxy clustering measurement based on the idea from [Johnston, et al](https://arxiv.org/abs/2012.08467).

To measure the two-point function of galaxy distribution, one needs to calculate so-called random-random and data-random pair counts (see https://ui.adsabs.harvard.edu/abs/1993ApJ...412...64L/abstract). However, if the galaxy sample has variable depth due to anisotropic selection effects by various systematics (Galactic extinction, seeing, etc), a uniform random will result in a biased estimation of the 2PCF. Therefore, one needs to recover the ''organized random'' to eliminate the selection effect of a synthesis of systematics. In practice, selection functions of systematics could be complicated thus cannot be fit with simple formula. In addition, different systematics might be correlated, which makes the problem more complicated.

The method proposed by [Johnston, et al](https://arxiv.org/abs/2012.08467) applies a combination of self-organizing map (SOM) and hierarchical clustering (HC) to capture the clustering of galaxies on high-dimensional systematics space, and then resample them back to the survey footprint to recover the organized random. This method has been tested on mock galaxy catalogs and the KiDS-1000 bright sample.

## Basic idea

The whole idea can be summarized as follows:

1. Train a SOM with systematics vectors of the galaxy sample;
2. Group the SOM weight with HC. Each group represents a subsample of galaxies that shares similar systematic and therefore selections;
3. Map galaxies from each hierarchical cluster back to the sky. Galaxies from one cluster occupies disjoint sky regions that share approximately the same selection function;
4. Resample galaxies randomly in each disjoint region according to the galaxy number of the corresponding hierarchical cluster;
5. Combine the resampled "galaxies" from all the clusters to get the organized random catalog.

In practice, the organized randoms are reconstructed as a pixelized weight map that quantifies the likelihood that a galaxie will be kept due to systematics selection in each part of the sky. In this package, we use the Healpix scheme to pixelize the organised randoms. We also pixelize the galaxy catalog to speed up 2PCF measurement.

## Mathematical notes

For each galaxy in a catalog, suppose that we have measured their systematics (for example, PSF\_FWHM, PSF\_ellipticity, magnitude limit, extinction, etc) which vary across the sky. Our goal is to find the total number of galaxies observed in each patch of the sky that might have been affected by these systematics. The idea is to first group these galaxies on high-dimensional systematics space, and assume that galaxies in each group are depleted uniformly. For each group, we find the sky regions that are occupied by galaxies from this group and the associated effective galaxy number density. Then we re-distribute galaxies uniformly in these regions. Finally, we combine those random galaxies from all the groups. 

To cluster systematics, we use the self-organising map (SOM) algorithm, which maps systematics vectors onto a 2D map while keeping the high-dimensional topology properties. Each cell on the 2D map corresponds to a subgroup of galaxies. Then we further group the SOM cells via hierarchical cluster (HC)

The effective pixel area (of the $p$-th pixel) occupied by galaxies from the $i$-th cluster is:

$$A_p^{i} \equiv \frac{N_p^{i}}{N_p}\times A_{p},$$

where $N_p$ is the total number of galaxies in the $p$-th pixel and $A_{p}$ is the observational footprint area in this pixel (we take into account the fractional coverage of some pixels).

Now we calculate the total effective area for each cluster by summing up all the occupied pixels:

$$A^{i} \equiv \sum_p A_p^{i},$$

and the effective surface number density of the $i$th cluster:
$$n_{i} \equiv \frac{N^{i}}{A^{i}}.$$

The organized-random can be thought of as randomly re-sampling galaxies in the regions occupied by each cluster number density given above, then combining all the clusters. Alternatively, one can also construct the ''organized random weight'' as:

$$w_p = \sum_{i} n_iA^i_p.$$
The organized random can be generated accordingly. Or alternatively, one can set this as the 'weight' parameter in the random catalog when calling [`treecorr`](https://rmjarvis.github.io/TreeCorr/_build/html/index.html) to measure galaxy correlation function.


## Structure of this package

All the source codes are stored in `./code/src`, including:

- OR_weights.py: the file that contains the class to recover organized random weight;
- treecorr_utils.py: contains a function to call `treecorr` to calculate 2PCF;
- plot_som.py: contains a function to make fancy SOM plots;
- glass_mock.py: generates mock catalogs with GLASS;
- generate_mocksys.py: assigns simple mock systematics and depletion functions to a galaxy sample.

In these files, only `OR_weights.py` is essential, and users are welcome to use their own codes to generate mock sample, calculate 2PCF and visualize them.

An example notebook is given in `./code/notebooks` which reads a mock catalog generated by the [GLASS](https://glass.readthedocs.io/en/stable/) package. Then it adds three types of systematics with simple depletion functions and train a SOM+HC to recover the organized random weight. $w(\theta)$'s' are calculated from un-depleted sample + uniform random (unbiased), depleted sample + uniform random (biased), depleted sample + depleted random (true OR), and depleted sample + reconstructed OR (recovered OR). If the idea works well, then $w(\theta)$ with the true OR and recovered OR should agree with each other.

## Required packages

Essential: `numpy`, `scipy`, `matplotlib`, `healpy`, [`somoclu`](https://somoclu.readthedocs.io/en/stable/) (to train a SOM), `sklearn` (to do hierarchical clustering)

Optional (for the example notebook): [`glass`](https://glass.readthedocs.io/en/stable/) (note that we use the version "2023.06") of glass-dev/glass here. other versions might not work well), [`treecorr`](https://rmjarvis.github.io/TreeCorr/_build/html/index.html), [`pyccl`](https://ccl.readthedocs.io/en/latest/index.html), `tqdm` (for showing progress bar)

## What is a "tiaogeng🥄"?

Tiaogeng (调羹) is the Chinese word for "spoon" which is more said in southern China. If we divide tiaogeng into two charactors, "tiao(调)" means "to reconcile" and "geng (羹)" means "Chinese-style thick soup". This is what this code is doing: handling the unevenly observed sky just like stirring your soup to make it taste more smooth and delicious.
