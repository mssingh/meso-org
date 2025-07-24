#
# A simple module to load in information about data locations etc.
# And include some simple functions to load datasets
#
# This allows the code to be a bit more portable
#

import xarray as xr
from glob import glob
from pathlib import Path



# Metadata ##########################################################

# Common time period
common_time=slice("2001-01-01", "2021-12-31")


# IMERG
IMERG = {
          "path" : "/g/data/k10/cb4968/metrics/observations/IMERG/small_domain/pr_based/IMERG/",
          "5deg" : dict(),
          "2deg" : dict(),
        }

IMERG["5deg"]["path"] =  IMERG["path"] + "pr_based_IMERG_3hrly_0-360_-30-30_3600x1800_2001-01_2023-12_boxsize_5/"
IMERG["5deg"]["files"] = sorted(glob(f"{IMERG['5deg']['path']}pr_based_IMERG_3hrly_*.nc"))

IMERG["2deg"]["path"] = IMERG["path"] + "pr_based_IMERG_3hrly_0-360_-30-30_3600x1800_2001-01_2001-2_boxsize_2/"
IMERG["2deg"]["files"] = sorted(glob(f"{IMERG['2deg']['path']}pr_based_IMERG_3hrly_*.nc"))


# ERA5
ERA5 = {
          "path" : "/g/data/k10/cr7888/",
          "5deg" : dict(),
          "2deg" : dict(),
}

ERA5["5deg"]["path"] = ERA5["path"]+"era5_daily_means_5deg/"
ERA5["5deg"]["lsm"]  = ERA5["5deg"]["path"]+"lsm_5deg.nc"

parent_dir = Path(ERA5["5deg"]["path"])
subdirs= [d for d in parent_dir.iterdir() if d.is_dir()]
ERA5["5deg"]["vars"] = [d.name for d in subdirs] 


ERA5["2deg"]["path"] = ERA5["path"]+"era5_daily_means_2deg/"
ERA5["2deg"]["lsm"]  = ERA5["2deg"]["path"]+"lsm_2deg.nc"

parent_dir = Path(ERA5["2deg"]["path"])
subdirs= [d for d in parent_dir.iterdir() if d.is_dir()]
ERA5["2deg"]["vars"] = [d.name for d in subdirs] 



# Utilities to read the data #############################################



def get_IMERG_data(grid="5deg",time=common_time,**kwargs):
    '''
    Read the IMERG data into an xarray dataset.

    ds = get_IMERG_data(grid="5deg",chunks={"time":1})

    grid = {"5deg" | "2deg"}        string describing the resolution grid
    any other keyword arguments will be passed to xr.open_mfdataset

    ds = xarray dataset
    '''

    # Read the data from the given grid
    ds = xr.open_mfdataset(IMERG[grid]["files"],**kwargs)

    # Restrict it to the common time
    ds = ds.sel(time=time)

    return ds






def get_ERA5_data(variables,grid="5deg",time=common_time,**kwargs):
    '''
    Read the ERA5 data for multiple variables and combine into a single
    xarray dataset.

    ds = get_ERA5_data(variables,grid="5deg")

    variables = ['t','u','v',...]   list of variabe strings required
    grid = {"5deg" | "2deg"}        string describing the resolution grid
    any other keyword arguments will be passed to xr.open_mfdataset
    
    ds = xarray dataset
    '''

    # Get the paths to each variable
    paths = [ERA5[grid]["path"]+f"/{var}/*.nc" for var in variables]


    # Open the datasets
    datasets = [xr.open_mfdataset(glob(p), combine="by_coords",**kwargs) for p in paths]

    # Also get the landmask
    ds_lsm = xr.open_dataset(ERA5[grid]["lsm"])

    datasets.append(ds_lsm)

    # Merge the datasets
    ds = xr.merge(datasets)

    # Restrict to the common time
    ds = ds.sel(time=time)

    # Make the time match that of IMERG
    ds["time"] = ds["time"].dt.floor("D")


    return ds


    





