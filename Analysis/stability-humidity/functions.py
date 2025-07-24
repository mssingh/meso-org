import numpy as np
from scipy.stats import binned_statistic_2d


def hist_2d_stat(x, y, z, x_edges=None, y_edges=None, counts_thresh=10):
    '''
    compute a 2d histogram of data density based on the input x and y
    and get the stats of the 2d histogram based on z (environmental conditions in our case)
    '''

    # create some bin edges if none are provided
    if x_edges is None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        x_edges = np.linspace(xmin,xmax, num=40, endpoint=true)

    if y_edges is None:
        ymin, xmax = np.nanmin(y), np.nanmax(y)
        y_edges = np.linspace(ymin,ymax, num=40, endpoint=true)

    ## mask out nans
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x2, y2, z2 = x[mask], y[mask], z[mask]
    
    stat, xedges, yedges, _ = binned_statistic_2d(
        x2,y2,z2,
        statistic='mean',
        bins=[x_edges, y_edges],
    )
    
    counts, _, _, _ = binned_statistic_2d(
        x2, y2, None, statistic='count',
        bins=[x_edges, y_edges],
    )
    
    ## mask out bins with less than the threshold counts
    stat[counts <= counts_thresh] = np.nan

    ## return transpose for plotting
    return {"x": xedges, "y": yedges, "stat": stat,"counts":counts}
 
