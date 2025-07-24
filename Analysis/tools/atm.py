import numpy as np


def e_sat(T):

      
        esl = c.e0*(T/c.T0)**((c.cpv-c.cpl)/c.Rv)*\
             np.exp( ( c.Lv0 - c.T0*(c.cpv-c.cpl) )/c.Rv * ( 1/c.T0 - 1/T ) )

        esi = c.e0*(T/c.T0)**((c.cpv-c.cpi)/c.Rv)*\
             np.exp( ( c.Ls0 - c.T0*(c.cpv-c.cpi) )/c.Rv * ( 1/c.T0 - 1/T ) )


        fice,fliq = calculate_frac_ice(T)
    
        es = esl*(fliq) + esi*fice

        return es

def calculate_frac_ice(T):

    
        fice = -( T-(c.T0) )/(c.deltaT)
        fice = np.where(fice<0,0,fice)
        fice = np.where(fice>1,1,fice)
        fliq = 1-fice

        return fice,fliq


def r_sat(T,p):

    
        es = e_sat(T);

        rs = c.eps * es/(p-es);

        return rs


def q_sat(T,p,qt=None):
    ''' 
    qs = q_sat(T,p [,qt,type,ice,deltaT])
    qs = saturation specific humidity (kg/kg)
    T = temperature (K)
    p = pressure (Pa)
    qt = mass fraction of total water
    '''

    # Calculate q_sat assuming no liquid/solid water
    es = e_sat(T)
    qs = c.eps*( es/(p-es*(1-c.eps)) )

    # Adjust for the existence of liquid/solid water
    if  qt is None:
       return qs

    if np.isscalar(T):
        if qs < qt:
            return (1-qt)*c.eps*( es/(p-es) )
        else:
            return qs

    return np.where(qs < qt,  (1-qt)*c.eps*( es/(p-es) ), qs)

def calculate_MSE(T,q,z):

    h = c.cp*T + c.g*z + c.Lv0*q

    return h

def calculate_MSE_sat(T,p,z):


    # Calculate saturation humidity
    qs = q_sat(T,p)
    
    h_sat = c.cp*T + c.g*z + c.Lv0*qs

    return h_sat

def calculate_RH(T,p,q):


    e = p*( q/(c.eps*(1-q)+q) )
    es = e_sat(T)
    
    # RH
    return e/es



class constants(object):

    def __init__(self):
   
        self.ice = 1                    # include ice?
        self.deltaT = 40                # mixed-phase range (K)

        ## Physical constants (based on constants in CM1)
        # Isobaric heat capacities (J/kg/K)
        self.cp        = 1005.7         # dry air
        self.cpv       = 1870.0         # water vapor
        self.cpl       = 4190.0         # liquid water
        self.cpi       = 2106.0         # solid water

        # Gas constants (J/kg/K)
        self.Rd        = 287.04         # dry air
        self.Rv        = 461.5

        # Latent heats (J/kg)
        self.Lv0       = 2501000.0      # liquid-gas
        self.Ls0       = 2834000.0      # solid-gas

        # gravitational acceleration (m/s^2)
        self.g         = 9.81

        # Liquid water density (kg/m^3)
        self.rhol = 1000

        ## Reference values
        self.T0        = 273.16                 # temperature (K)
        self.p00       = 100000            # Pressure (Pa)
        self.e0        = 611.2          # vapor pressure (Pa)

        ## Derived parameters
        self.cv        = self.cp-self.Rd
        self.cvv       = self.cpv-self.Rv
        self.eps       = self.Rd/self.Rv


# Load the thermodynamics constants
c = constants()
