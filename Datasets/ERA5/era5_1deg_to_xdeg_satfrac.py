#!usr/bin/env python3

"""
Saturation fraction is treated differently, since it requires 3D fields of q and T to calculate
"""

import xarray as xr
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import warnings


R_d = 287                      # dry gas constant for earth's atmopshere
R_v = 461                      # gas cosntant for water vapour
eps = R_d/R_v                  # ratio of dry/moist gas constants


def saturation_vapour_pressure(T):
    """
    Bolton formula
    """
    return 611.2 * np.exp((17.67 * (T - 273.15)) / (T - 29.65))


def spec_hum(p, e):
    """
    converts vapour pressure to specific humidity
    """
    r = (eps * e) / (p - e)
    return r / (1 + r)


def satfrac_coarsen_and_save(res):

    for year in range(1979, 2022):
    
        print(year)
    
        # open 1 deg daily means
        q = xr.open_dataset(f'/g/data/k10/cr7888/era5_daily_means/qall/era5_q_daily_mean_{year}.nc')['q']
        T = xr.open_dataset(f'/g/data/k10/cr7888/era5_daily_means/tall/era5_t_daily_mean_{year}.nc')['t']
    
        # cut off the top levels since weird stuff happens there
        q = q.sel(level=slice(50, 1000))
        T = T.sel(level=slice(50, 1000))
    
        # change dim names if necessary
        if 'latitude' in q.dims:
            q = q.rename({'latitude': 'lat', 'longitude': 'lon'})
        if 'latitude' in T.dims:
            T = T.rename({'latitude': 'lat', 'longitude': 'lon'})
    
        # change lons to 0-360
        q['lon'] = q['lon'] % 360
        q = q.sortby(T.lon)
        T['lon'] = T['lon'] % 360
        T = T.sortby(T.lon)
    
        # best to calculate sat frac before coarsening
    
        es = saturation_vapour_pressure(T)
        qs = spec_hum(es.level*100, es)
        
        # sat frac = int(q) / int(qs), not int(q / qs)
        sat_frac = q.integrate('level') / qs.integrate('level')
    
        # coarsen (average) data to 2 or 5 deg grids
        sat_frac = sat_frac.interp(lat=np.arange(-29.5, 30.5), lon=np.arange(0.5, 360.5))
        sat_frac_coarsen = sat_frac.coarsen({'lat': res, 'lon': res}, boundary='trim').mean()
        sat_frac_coarsen = sat_frac_coarsen.to_dataset(name='satfrac')
        
        ## save outputs
        sat_frac_coarsen.to_netcdf(f'/g/data/k10/cr7888/era5_daily_means_{res}deg/satfrac/era5_satfrac_daily_mean_{res}deg_{year}.nc', 
                                    encoding={'satfrac': {'zlib': True, "complevel": 5}})
    
        # clear memory
        del q
        del T
    
        #break


if __name__ == "__main__":
    
    cluster = LocalCluster()
    client = Client(cluster)

    satfrac_coarsen_and_save(2)
    satfrac_coarsen_and_save(5)

