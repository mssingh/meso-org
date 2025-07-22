#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d


# In[2]:


params={     
    'axes.labelsize'  : '14',   
    'axes.titlesize'  : '14',  
    'xtick.labelsize' :'14',
    'ytick.labelsize' :'14',    
    'lines.linewidth' : '2' ,   
    'legend.fontsize' : '10', 
    'figure.figsize'   : '12, 7'    
}
plt.rcParams.update(params)


# In[3]:


## IMERG
satellite_path = "/g/data/k10/cb4968/metrics/observations/IMERG/small_domain/pr_based/IMERG/pr_based_IMERG_3hrly_0-360_-30-30_3600x1800_2001-01_2023-12_boxsize_5/"
sat_files = sorted(glob(f"{satellite_path}pr_based_IMERG_3hrly_*.nc"))
ds_sat = xr.open_mfdataset(sat_files,chunks={"time":10})
# ERA5
ds_vo500 = xr.open_mfdataset(sorted(glob("/g/data/k10/cr7888/era5_daily_means_5deg/vo500/*.nc")))
ds_shear = xr.open_dataset("/g/data/k10/dl6968/meso_org_data/ERA5_5deg/wind_shear_500-1000hpa.nc")
## land sea mask
ds_lsm = xr.open_dataset("/g/data/k10/cr7888/era5_daily_means_5deg/lsm_5deg.nc") 


# In[4]:


lsm = ds_lsm["lsm"]
# max precipitation
max_pr = ds_sat["pr_based_max_pr_30min"].resample(time='1D').max().where(lsm == 0)
# mean precipitation
mean_pr = ds_sat["pr_based_mean_pr"].resample(time='1D').mean().where(lsm == 0)
# Iorg
iorg= ds_sat["pr_based_Iorg"].resample(time='1D').mean().where(lsm == 0)
## make ERA5 data timestamps to match IMERG data
ds_vo500["time"] = ds_vo500["time"].dt.floor("D")
ds_shear["time"] = ds_shear["time"].dt.floor("D")


# In[5]:


get_ipython().run_cell_magic('time', '', '## IMERG daily to match ERA5 timestamps\nmean_pr_match = mean_pr.sel(time=slice(mean_pr["time"][0],ds_shear["time"][-1])).compute()\niorg_match = iorg.sel(time=slice(mean_pr["time"][0],ds_shear["time"][-1])).compute()\nmax_pr_match = max_pr.sel(time=slice(mean_pr["time"][0],ds_shear["time"][-1])).compute()     \n')


# In[6]:


get_ipython().run_cell_magic('time', '', '## ERA5 vorticity\nvo500 = ds_vo500["vo"].sel(time=slice(mean_pr["time"][0],ds_shear["time"][-1])).where(lsm == 0).compute()\n# shear\nspeed_shear = ds_shear["vertical_shear"].sel(time=slice(mean_pr["time"][0],ds_shear["time"][-1])).where(lsm == 0).compute()\nwdir_shear = ds_shear["direction_shear"].sel(time=slice(mean_pr["time"][0],ds_shear["time"][-1])).where(lsm == 0).compute()\n')


# In[7]:


def compute_2d_hist(x, y, xmin=0, xmax=7, ymin=None, ymax=None, nbins=50, counts_thresh=10):
    '''
    Compute a 2D histogram of data density based on the input x and y
    '''
    
    if ymin==None and ymax==None:
        ymin, ymax = np.nanmin(y), np.nanmax(y)
    # 1) compute the raw 2D histogram
    counts, xedges, yedges = np.histogram2d(
        x, y,
        bins=nbins,
        range=[[xmin,xmax], [ymin, ymax]]
    )
    
    # 2) mask out bins with less than the threshold counts
    counts_masked = np.where(counts > counts_thresh, counts, np.nan)
    
    # 3) build the mesh to plot
    X, Y = np.meshgrid(xedges, yedges)
    return X, Y, counts_masked.T


# In[8]:


def compute_2d_hist_stat(x, y, z, xmin=0, xmax=7, ymin=None, ymax=None, nbins=50, counts_thresh=10):
    '''
    Compute a 2D histogram of data density based on the input x and y
    And get the stats of the 2D histogram based on z (environmental conditions in our case)
    '''
    if ymin==None and ymax==None:
        ymin, ymax = np.nanmin(y), np.nanmax(y)

    xbins, ybins = nbins, nbins
    xrange = [xmin, xmax]
    yrange = [ymin, ymax]

    ## mask out nans
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x2, y2, z2 = x[mask], y[mask], z[mask]
    
    stat, xedges, yedges, _ = binned_statistic_2d(
        x2,y2,z2,
        statistic='mean',
        bins=[xbins, ybins],
        range=[xrange, yrange]
    )
    
    counts, _, _, _ = binned_statistic_2d(
        x2, y2, None, statistic='count',
        bins=[xbins, ybins],
        range=[xrange, yrange]
    )
    
    ## mask out bins with less than the threshold counts
    stat[counts <= counts_thresh] = np.nan
    ## return transpose for plotting
    return {"x": xedges, "y": yedges, "stat": stat.T}
    


# In[9]:


mean_pr_1D = mean_pr_match.values.flatten()
iorg_1D = iorg_match.values.flatten()
max_pr_1D = max_pr_match.values.flatten()
vo500_1D = vo500.values.flatten()
speed_shear_1D = speed_shear.values.flatten()
wdir_shear_1D = wdir_shear.values.flatten()


# In[19]:


## Vorticity at 500 hPa
mean_max_vo500 = compute_2d_hist_stat(mean_pr_1D, max_pr_1D, vo500_1D, nbins=60, counts_thresh=10)
mean_iorg_vo500 = compute_2d_hist_stat(mean_pr_1D, iorg_1D, vo500_1D, nbins=60, counts_thresh=10)
## Vertical shear 500-1000 hPa
mean_max_speed = compute_2d_hist_stat(mean_pr_1D, max_pr_1D, speed_shear_1D, nbins=60, counts_thresh=10)
mean_iorg_speed = compute_2d_hist_stat(mean_pr_1D, iorg_1D, speed_shear_1D, nbins=60, counts_thresh=10)
## Direction shear 500-1000 hPa
mean_max_wdir = compute_2d_hist_stat(mean_pr_1D, max_pr_1D, wdir_shear_1D, nbins=60, counts_thresh=10)
mean_iorg_wdir = compute_2d_hist_stat(mean_pr_1D, iorg_1D, wdir_shear_1D, nbins=60, counts_thresh=10)


# In[20]:


fig = plt.figure(figsize=(12, 12))

## mean vs max
### vorticity
ax1 = fig.add_subplot(321)
pcm1 = ax1.pcolormesh(
    mean_max_vo500["x"], mean_max_vo500["y"], mean_max_vo500["stat"],
    cmap='RdYlBu_r', shading='auto', vmin=-1*10**(-5), vmax = 10**(-5)
)
cbar = plt.colorbar(pcm1, ax=ax1, label="Vorticity (s$^{-1}$)",extend="both")
ax1.set_xlabel("Mean precipitation (mm h$^{-1}$)")
ax1.set_ylabel("Max precipitation (mm h$^{-1}$)")
ax1.set_xlim([0,7])
ax1.text(6,180, "(a)",fontsize=14, fontweight="bold") # x, y, s


### vertical shear
ax2 = fig.add_subplot(323)
pcm2 = ax2.pcolormesh(
    mean_max_speed["x"], mean_max_speed["y"], mean_max_speed["stat"],
    cmap='RdYlBu_r', shading='auto',vmin=5, vmax=20
)
cbar = plt.colorbar(pcm2, ax=ax2, label="Vertical shear (m s$^{-1}$)",extend="both")
ax2.set_xlabel("Mean precipitation (mm h$^{-1}$)")
ax2.set_ylabel("Max precipitation (mm h$^{-1}$)")
ax2.set_xlim([0,7])
ax2.text(6,180, "(c)",fontsize=14, fontweight="bold") # x, y, s

### direction shear
ax3 = fig.add_subplot(325)
pcm3 = ax3.pcolormesh(
    mean_max_wdir["x"], mean_max_wdir["y"], mean_max_wdir["stat"],
    cmap='RdYlBu_r', shading='auto',vmin=-50, vmax=50,
)
cbar = plt.colorbar(pcm3, ax=ax3, label="Directional shear (degrees)",extend="both")

ax3.set_xlabel("Mean precipitation (mm h$^{-1}$)")
ax3.set_ylabel("Max precipitation (mm h$^{-1}$)")
ax3.set_xlim([0,7])
ax3.text(6,180, "(e)",fontsize=14, fontweight="bold") # x, y, s



## mean vs Iorg
### vorticity
ax4 = fig.add_subplot(322)
pcm4 = ax4.pcolormesh(
    mean_iorg_vo500["x"], mean_iorg_vo500["y"], mean_iorg_vo500["stat"],
    cmap='RdYlBu_r', shading='auto',vmin=-1*10**(-5), vmax = 10**(-5)
)
cbar = plt.colorbar(pcm4, ax=ax4, label="Vorticity (s$^{-1}$)",extend="both")
ax4.set_ylabel("I$_{org}$")
ax4.set_xlabel("Mean precipitation (mm h$^{-1}$)")
ax4.set_xlim([0,7])
ax4.text(6,0.85, "(b)",fontsize=14, fontweight="bold") # x, y, s


### vertical shear
ax5 = fig.add_subplot(324)
pcm5 = ax5.pcolormesh(
    mean_iorg_speed["x"], mean_iorg_speed["y"], mean_iorg_speed["stat"],
    cmap='RdYlBu_r', shading='auto',vmin=5, vmax=20
)
cbar = plt.colorbar(pcm5, ax=ax5, label="Vertical shear (m s$^{-1}$)",extend="both")
ax5.set_ylabel("I$_{org}$")
ax5.set_xlabel("Mean precipitation (mm h$^{-1}$)")
ax5.set_xlim([0,7])
ax5.text(6,0.85, "(d)",fontsize=14, fontweight="bold") # x, y, s

### vertical shear
ax6 = fig.add_subplot(326)
pcm6 = ax6.pcolormesh(
    mean_iorg_wdir["x"], mean_iorg_wdir["y"], mean_iorg_wdir["stat"],
    cmap='RdYlBu_r', shading='auto', vmin=-50, vmax=50,
)
cbar = plt.colorbar(pcm6, ax=ax6, label="Directional shear (degrees)",extend="both")
ax6.set_ylabel("I$_{org}$")
ax6.set_xlabel("Mean precipitation (mm h$^{-1}$)")
ax6.set_xlim([0,7])
ax6.text(6,0.85, "(f)",fontsize=14, fontweight="bold") # x, y, s

plt.tight_layout()

plt.savefig("/home/565/dl6968/meso-org/Figures/Fig7.png", dpi=200, bbox_inches="tight")
plt.show()


# In[ ]:




