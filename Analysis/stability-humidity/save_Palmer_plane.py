'''
Script to plot metrics of the precipitation distribution in 
the stability-humidity phase plane.

The phase plane is described in Palmer & Singh (2024).


'''

# Set debugging mode
debug = 1


# Modules #########################################################

import sys
sys.path.append("/g/data/wa6/mss565/Analysis/meso-org/Analysis/tools")

# utilities
from glob import glob

# Data modules
import xarray as xr
import numpy as np



# Local modules
import util
import functions

# Parameters #######################################################

## Physics

# grid resolution
grid = "5deg"

# land fraction that we call ocean
lfrac = 0.25


# Pressure levels to evaluate phase plane
p_lower = 85000
p_upper = 50000

# precipition threshold for convection
pr_thresh = 5/24 # mm/hr

## Technical

# Chunking for satellite data
Nchunk = 8


# Load the data ################################################

if debug==1:
   time_period = slice("2001-01-01", "2001-01-31")
else:
   time_period = util.common_time


# Load the satellite data
print("Loading satellite data...")
ds_sat = util.get_IMERG_data(grid=grid,time=time_period,chunks={"time":Nchunk})

# Load the ERA5 data
print("Loading ERA5 data...")
variables = ["t","satfrac"]
ds_ERA5 = util.get_ERA5_data(variables,grid=grid,time=time_period)
ds_ERA5 = ds_ERA5.chunk({'time': Nchunk})

# Convert to Pa
ds_ERA5["level"] = ds_ERA5["level"]*100

# get the landmask
lsm = ds_ERA5["lsm"]

## Resample to daily over the ocean
print('Matching data times...')

# Calculate daily maximum precipitation
max_pr = ds_sat["pr_based_max_pr_30min"].resample(time='1D').max().where(lsm <= lfrac)

# Resmaple everything else by mean
ds_sat = ds_sat.resample(time='1D').max().where(lsm <= lfrac)

# Add the maximum precip back
ds_sat["pr_based_max_pr_30min"] = max_pr



# Calculate the phase-plane variables #############################################


## calculate environments

# Humidity and height data missing, so use temp for now
dMSEs = ds_ERA5["t"].sel(level=p_upper) -  ds_ERA5["t"].sel(level=p_lower)
satdef = ds_ERA5["satfrac"]

# Flatten The precipitation data
mean_pr = ds_sat.pr_based_mean_pr.values.flatten()
Iorg = ds_sat.pr_based_Iorg.values.flatten()
max_pr = ds_sat.pr_based_max_pr_30min.values.flatten()
mean_area = ds_sat.pr_based_mean_area.values.flatten()

# Flatten the environments
dMSEs = dMSEs.values.flatten()
satdef = satdef.values.flatten()

# Remove the non-rainy points
I = mean_pr>pr_thresh

mean_pr = mean_pr[I]
Iorg = Iorg[I];
max_pr = max_pr[I];
mean_area = mean_area[I];

dMSEs = dMSEs[I];
satdef = satdef[I];


# Make the histogram ########################################################
satdef_edges = np.linspace(0.5, 1, num=26, endpoint=True)
dMSEs_edges = np.linspace(-25, -21, num=41, endpoint=True)

satdef_bins = (satdef_edges[0:-1]+satdef_edges[1:])/2
dMSEs_bins = (dMSEs_edges[0:-1]+dMSEs_edges[1:])/2

mean_pr_hist = functions.hist_2d_stat(satdef, dMSEs, mean_pr, x_edges=satdef_edges, y_edges=dMSEs_edges, counts_thresh=10)
mean_area_hist = functions.hist_2d_stat(satdef, dMSEs, mean_area, x_edges=satdef_edges, y_edges=dMSEs_edges, counts_thresh=10)
Iorg_hist = functions.hist_2d_stat(satdef, dMSEs, Iorg, x_edges=satdef_edges, y_edges=dMSEs_edges, counts_thresh=10)
max_pr_hist = functions.hist_2d_stat(satdef, dMSEs, max_pr, x_edges=satdef_edges, y_edges=dMSEs_edges, counts_thresh=10)


# Make the dataset #########################################################



# Create Dataset
ds = xr.Dataset(
    {
        "counts": (["sat_deficit", "stability"], mean_pr_hist["counts"]),
        "mean_pr": (["sat_deficit", "stability"], mean_pr_hist["stat"]),
        "max_pr": (["sat_deficit", "stability"], max_pr_hist["stat"]),
        "Iorg": (["sat_deficit", "stability"], Iorg_hist["stat"]),
        "mean_area": (["sat_deficit", "stability"], mean_area_hist["stat"]),
    },
    coords={
        "sat_deficit": satdef_bins,
        "stability": dMSEs_bins,
        "sat_deficit_edges": satdef_edges,
        "stability_edges": dMSEs_edges,
    },
)

ds["sat_deficit"].attrs["long name"] = "saturation deficit"
ds["sat_deficit"].attrs["units"] = "J kg^{-1}"
ds["sat_deficit"].attrs["formula"] = "int{ Lv(q*-q) dz }"

ds["stability"].attrs["long name"] = "stability index"
ds["stability"].attrs["units"] = "J kg^{-1}"
ds["stability"].attrs["formula"] = "Delta{ MSE* }"

ds["counts"].attrs["units"] = " "
ds["counts"].attrs["long name"] = "number of occurences "

ds["mean_pr"].attrs["units"] = "mm hr^{-1}"
ds["mean_pr"].attrs["long name"] = "mean precipitation"

ds["max_pr"].attrs["units"] = "mm hr^{-1}"
ds["max_pr"].attrs["long name"] = "maximum precipitation"

ds["mean_area"].attrs["units"] = "km^{2}"
ds["mean_area"].attrs["long name"] = "mean area of objects"

ds["Iorg"].attrs["units"] = " "
ds["Iorg"].attrs["long name"] = "Index of organisation"



ds.attrs["grid_resolution"] = grid
ds.attrs["domain"] = "30S-30N"
ds.attrs["surface"] = "ocean only"

ds["precip_thresh"] = xr.DataArray(pr_thresh)
ds["precip_thresh"].attrs["long_name"] = "precipitation threshold for convection"
ds["precip_thresh"].attrs["units"] = "mm hr^{-1}"

ds["pressure_lower"] = xr.DataArray(p_lower)
ds["pressure_lower"].attrs["long_name"] = "pressure at lower boundary of layer"
ds["pressure_lower"].attrs["units"] = "Pa"

ds["pressure_upper"] = xr.DataArray(p_upper)
ds["pressure_upper"].attrs["long_name"] = "pressure at upper boundary of layer"
ds["pressure_upper"].attrs["units"] = "Pa"



ds.to_netcdf("./processed_data/Palmer_plane_ocean_30S-30N.nc")



# Plot the histogram #######################################################











 
