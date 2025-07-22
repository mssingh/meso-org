#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


# In[2]:


## ERA5
ds_u = xr.open_mfdataset(sorted(glob("/g/data/k10/cr7888/era5_daily_means_5deg/u/*.nc")))
ds_v = xr.open_mfdataset(sorted(glob("/g/data/k10/cr7888/era5_daily_means_5deg/v/*.nc")))


# In[4]:


u500 = ds_u["u"].sel(level=500)
u1000 = ds_u["u"].sel(level=1000)

v500 = ds_v["v"].sel(level=500)
v1000 = ds_v["v"].sel(level=1000)


# In[8]:


## ChatGPT made my old wind direction function fancy
def calc_wdir_xr(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Calculate wind direction from u and v components using xarray,
    returning direction in degrees from which the wind is blowing (0° = north).
    """

    wdir = np.arctan2(u, v) * 180 / np.pi
    wdir = xr.where(wdir < 0, wdir + 360, wdir)

    # Optionally assign metadata
    wdir.name = "wind_direction"
    wdir.attrs["units"] = "degrees"
    wdir.attrs["description"] = "Meteorological wind direction (0° = north, 90° = east)"

    return wdir


# In[9]:


wdir500 = calc_wdir_xr(u500, v500)
wdir1000 = calc_wdir_xr(u1000, v1000)


# In[10]:


vertical_shear = np.sqrt( (u500-u1000)**2 + (v500-v1000)**2 )
wdir_shear = wdir500-wdir1000


# In[12]:


get_ipython().run_cell_magic('time', '', 'vertical_shear = vertical_shear.compute()\n')


# In[18]:


vertical_shear.attrs["units"] = "m/s"
vertical_shear.attrs["description"] = "wind speed shear between 500 hPa and 1000 hPa"


# In[14]:


get_ipython().run_cell_magic('time', '', 'wdir_shear = wdir_shear.compute()\n')


# In[21]:


wdir_shear.attrs["units"] = "m/s"
wdir_shear.attrs["description"] = "wind direction shear between 500 hPa and 1000 hPa"


# In[22]:


ds_shear = xr.Dataset({
    "direction_shear": wdir_shear,
    "vertical_shear": vertical_shear
})


# In[23]:


## save it to netcdf 
ds_shear.to_netcdf("/g/data/k10/dl6968/meso_org_data/ERA5_5deg/wind_shear_500-1000hpa.nc")


# In[24]:


ds_u.close()
ds_v.close()
ds_shear.close()


# In[ ]:




