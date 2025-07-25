
import xarray as xr
import numpy as np
# plotting modules
import matplotlib.pyplot as plt


# Load the dataset
ds = xr.open_dataset("./processed_data/Palmer_plane_ocean_30S-30N.nc")

ds["eff_rad"] = ds["mean_area"]**0.5;



def plot_plane(ds,var,the_title):

    Cmax = np.max(ds.counts)
    plot_var= np.where(ds.counts < Cmax/10,np.nan, ds[var].values)

    fig = plt.figure(figsize=(5, 7))

    # Plot counts
    ax1 = fig.add_axes([0.2, 0.55, 0.7, 0.4])
    pcm1 = plt.pcolormesh(
        ds.sat_deficit_edges, ds.stability_edges, ds.counts.T,    
        cmap='Blues',
        shading='auto'
    )
    plt.colorbar(pcm1, label="Count",extend ="max")
    plt.ylabel("stability (kJ kg$^{-1}$)")
    ax1.set_xlim([0,12])
    ax1.set_ylim([-12, 2])

    # Plot stat
    ax1 = fig.add_axes([0.2, 0.1, 0.7, 0.4])
    pcm1 = plt.pcolormesh(
        ds.sat_deficit_edges, ds.stability_edges, plot_var.T,    
        cmap='YlOrRd',
        shading='auto'
    )
    levs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    levs = levs*Cmax.item()
    contours = ax1.contour(ds.sat_deficit, ds.stability, ds.counts.T, levels=levs, colors='black', linewidths=1)
    plt.colorbar(pcm1, label=the_title,extend ="max")
    plt.xlabel("saturation deficit (kJ kg$^{-1}$)")
    plt.ylabel("stability (kJ kg$^{-1}$)")
    ax1.set_xlim([0,12])
    ax1.set_ylim([-12, 2])

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

tit = "eff. radius (km)"
var = "eff_rad"

plot_plane(ds,var,tit)
