import fitsio
from astropy.io import fits
import numpy as np
from minisom import MiniSom
import matplotlib as mpl
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import RegularPolygon, Ellipse
from matplotlib import cm, colorbar

mpl.rcParams['figure.dpi'] = 200
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_som(ax, som_heatmap, topology='rectangular', colormap=cm.viridis, cbar_name=None,
            vmin=None, vmax=None):
    '''
    This function plots the pre-trained SOM.
    Input:
    ax: the axis to be plotted on.
    som_heatmap: a 2-D array contains the value in a pre-trained SOM. The value can be the number 
    of sources in each cell; or the mean feature in every cell.  
    topology: string, either 'rectangular' or 'hexagonal'.
    colormap: the colormap to show the values. default: cm.viridis.
    cbar_name: the label on the color bar.
    '''
    if vmin == None and vmax == None:
        vmin = np.quantile(som_heatmap[~np.isnan(som_heatmap)],0.01)
        vmax = np.quantile(som_heatmap[~np.isnan(som_heatmap)],0.99)
    cscale = (som_heatmap-vmin) / (vmax - vmin)
    som_dim = cscale.shape[0]
    if topology == 'rectangular':
        ax.matshow(som_heatmap.T, cmap=colormap, 
                   vmin=vmin, 
                   vmax=vmax)
    else:
                
        yy, xx= np.meshgrid(np.arange(som_dim), np.arange(som_dim))
        shift = np.zeros(som_dim)
        shift[::2]=-0.5
        xx = xx + shift
        for i in range(cscale.shape[0]):
            for j in range(cscale.shape[1]):
                wy = yy[(i, j)] * np.sqrt(3) / 2
                if np.isnan(cscale[i,j]):
                    color = 'k'
                else:
                    color = colormap(cscale[i,j])
            
                hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius= 1 / np.sqrt(3),
                                 facecolor=color, 
                                 edgecolor=color,
                                 #alpha=.4, 
                                 lw=0.2,)
                ax.add_patch(hex)

    scmap = plt.scatter([0,0],[0,0], s=0, c=[vmin, vmax], 
                            cmap=colormap)
    ax.set_xlim(-1,som_dim-.5)
    ax.set_ylim(-0.5,som_dim * np.sqrt(3) / 2)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
        
    cb = plt.colorbar(scmap, cax=cax)
    cb.ax.tick_params(labelsize=5)
    cb.set_label(cbar_name, size='xx-small') 
    ax.axis('off')