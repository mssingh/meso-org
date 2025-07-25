#!usr/bin/env python3

import xarray as xr
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import warnings


def coarsen_and_save(var, res, invar, method='mean'):
    # 'invar' is for the input variables in current 1 deg daily means folder (this may differ from "var",
    # e.g. I have called u on pressure levels "uall" in my 1 deg folder)
    # 'method' is mean by default but can also be set to "min" or max" to get the max value within the mesoscale domain
    # (e.g. max CAPE)

    for year in range(1979, 2022):

        print(year)

        # open 1 deg daily means
        data = xr.open_dataset(f'/g/data/k10/cr7888/era5_daily_means/{invar}/era5_{var}_daily_mean_{year}.nc')[var]

        # change dim names if necessary
        if 'latitude' in data.dims:
            data = data.rename({'latitude': 'lat', 'longitude': 'lon'})
    
        # change lons to 0-360
        data['lon'] = data['lon'] % 360
        data = data.sortby(data.lon)
    
        # coarsen (average) data to 2 or 5 deg grids
        data = data.interp(lat=np.arange(-29.5, 30.5), lon=np.arange(0.5, 360.5))
        if method == 'mean':
            data_coarsen = data.coarsen({'lat': res, 'lon': res}, boundary='trim').mean()
        elif method == 'min':
            data_coarsen = data.coarsen({'lat': res, 'lon': res}, boundary='trim').min()
        elif method == 'max':
            data_coarsen = data.coarsen({'lat': res, 'lon': res}, boundary='trim').max()
        
        ## save outputs
        data_coarsen.to_netcdf(f'/g/data/k10/cr7888/era5_daily_means_{res}deg/{var}/era5_{var}_daily_mean_{res}deg_{year}.nc', 
                                    encoding={var: {'zlib': True, "complevel": 5}})

        # clear memory
        del data        
        del data_coarsen
        
        #break


if __name__ == "__main__":
    
    cluster = LocalCluster()
    client = Client(cluster)

    # put the variables and resolutions you want here

    coarsen_and_save('vo', 2, 'vo500')
    coarsen_and_save('vo', 5, 'vo500')

