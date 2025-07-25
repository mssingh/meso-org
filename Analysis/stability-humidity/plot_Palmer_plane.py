
import xarray as xr
import numpy as np
# plotting modules
import matplotlib.pyplot as plt


# Load the dataset
ds = xr.open_dataset("./processed_data/Palmer_plane_ocean_30S-30N.nc")

ds["eff_rad"] = ds["mean_area"]**0.5;



def plot_plane(ds,var,the_title):


    plot_var= np.where(ds.counts < 1000,np.nan, ds[var].values)

    fig = plt.figure(figsize=(5, 7))

    # Plot counts
    ax1 = fig.add_axes([0.2, 0.55, 0.7, 0.4])
    pcm1 = plt.pcolormesh(
        ds.sat_deficit_edges, ds.stability_edges, ds.counts.T,    
        cmap='Blues',
        shading='auto'
    )
    plt.colorbar(pcm1, label="Count",extend ="max")
    plt.ylabel("$\Delta T$ (K)")
    ax1.set_xlim([0.5,1])
    ax1.set_ylim([-25, -21])

    # Plot stat
    ax1 = fig.add_axes([0.2, 0.1, 0.7, 0.4])
    pcm1 = plt.pcolormesh(
        ds.sat_deficit_edges, ds.stability_edges, plot_var.T,    
        cmap='YlOrRd',
        shading='auto'
    )
    plt.colorbar(pcm1, label=the_title,extend ="max")
    plt.xlabel("saturation fraction")
    plt.ylabel("$\Delta T$ (K)")
    ax1.set_xlim([0.5,1])
    ax1.set_ylim([-25, -21])

    fig.savefig('./figures/Palmer_plane_'+var+'.pdf')
    fig.show()



tit = "mean precip. (mm hr$^{-1}$)"
var = "mean_pr"



plot_plane(ds,var,tit)

tit = "object area. (km$^{2}$)"
var = "mean_area"

plot_plane(ds,var,tit)

tit = "max precip. (mm hr$^{-1}$)"
var = "max_pr"

plot_plane(ds,var,tit)

tit = "eff. radius (km))"
var = "eff_rad"

plot_plane(ds,var,tit)