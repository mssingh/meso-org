

def calculate_stability_index(T,z,zu,pl,pu):

    # Import thermodynamic functions 
    import atm


    hsl = calculate_MSE_sat(Tl,pl,zl)
    hsu = calculate_MSE_sat(Tu,pl,zl)
    
    # Stability index
    dh = hsu-hsl

    return dh



def calculate_saturation_deficit(T,q,p_lower,p_upper):
## get saturation deficit

    # Get the variables we need at each level
    ds = ds.sel(pressure=slice(p_upper,p_lower))

    # Saturation humidity
    qs = calculate_saturation_humidity(ds)

    # Saturation deficit
    dq = c.Lv0*(qs-ds["hus"])

    ## Integrate

    # This will keep dqint as an xarrary
    dqint = dq.sel(pressure=p_upper)*0.0

    # Now integrate in height
    dqint = dqint - np.trapezoid(dq,x=ds.zg,axis=0)


    return dqint




