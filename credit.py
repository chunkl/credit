import numpy as np
import timeit
import numpy as np
import matplotlib.pyplot as plt
import timeit
import multiprocessing
import matplotlib as mpl
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.colors as colors
from scipy.integrate import tplquad, quad, quad_vec, dblquad
from datetime import datetime
from cycler import cycler
from scipy.special import iv
from scipy import interpolate

#time the code
start = timeit.default_timer()
startt = datetime.now()
c = 3e10 #cm/s
M_PI = 134.9768e-6 #TeV
M_PROTON = 938.27208816e-6 #TeV
T_TH = 2*M_PI + M_PI**2 / (2*M_PROTON)
YEAR_SECOND_CONVERSION = 31557600
CM_PC_CONVERSION = 3.0857e18
Earray = np.logspace(-2,5,1000)
Ea = np.logspace(-3,np.log10(3e3),100)
#define constants
PROTON_INJECTION_SPECTRUM = 2.2
DIFFUSION_SPECTRUM = 0.6
ESCAPE_MOMENTUM_SPECTRUM = 2.5
DIFFUSION_SUPPRESSION_FACTOR = 0.1
DIFFUSION_COEFFICIENT = 6.7825E27
ISM_PARTICLE_DENSITY = 1
MAXIMUM_PARTICLE_MOMENTUM = 3E3 #TeV
SEDOV_SNR_II = 1.6E3
SEDOV_SNR_1A = 234
MML_L = 92.272
MML_B = 2.775
MML_D = 3.28E3
MML_DENSITY = 30
MML_DIAMETER = 0.5
FKT_L = 92.4
FKT_B = 3.2
FKT_D = 1.7E3
FKT_DENSITY = 37
FKT_DIAMETER = 1.1
SEDOV_TIME = 1e3
SOURCE_LIFETIME = 1e5
MAXIMUM_RIGIDITY = 3e3
MINIMUM_RIGIDITY = 1e-3
GALACTIC_RADIUS = 15000
HALO_HEIGHT = 4000
SUN_POSITION = 8300


approx_AMS = np.sqrt(1.16)*1.10655**np.arange(72)
approx_DAMPE = 1983*1.58308693**np.arange(9)
AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY = np.concatenate((approx_AMS, approx_DAMPE))

conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
Dpcy = DIFFUSION_COEFFICIENT * conversion

def D(E, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    return D_0*(E*1000)**(delta)

def source_unnormalized_integrant(R):
    r_0 = SUN_POSITION
    result = 4.745 * np.exp(-(R - r_0)/4500)
    if R>3.7:
        result += 26.679 * np.exp(-(R**2 - r_0**2)/(6800**2))
    else:
        result += 26.679 * np.exp(-(3700**2 - r_0**2)/(6800**2)) * np.exp(-((R - 3700)/2100)**2)
    return 2*np.pi*R*result

sourcechange = quad(source_unnormalized_integrant,0,GALACTIC_RADIUS)[0]

def FerrierSource(R, r_0 = SUN_POSITION):
    change = sourcechange
    r_0 = SUN_POSITION
    result = 4.745 * np.exp(-(R - r_0)/4500)
    if R>3.7:
        result += 26.679 * np.exp(-(R**2 - r_0**2)/(6800**2))
    else:
        result += 26.679 * np.exp(-(3700**2 - r_0**2)/(6800**2)) * np.exp(-((R - 3700)/2100)**2)
    return result*3/100/change

def escape_time(E, p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY, t_break=SOURCE_LIFETIME-0.001):
    yrs = YEAR_SECOND_CONVERSION
    t_sed = SEDOV_TIME
    if p_break > p_max:
        return yrs * t_sed
    elif p_break < p_min:
        return yrs * t_sed * np.exp(1 - E/p_max)
    else:
        s1 = np.log(t_sed/t_break) / np.log(p_max/p_break)
        s2 = np.log(t_break/t_life) / np.log(p_break/p_min)
        return yrs*np.where(E>p_break, t_sed*(p_max/E)**-s1,t_break*(p_break/E)**-s2)

def source_near_sun_integrant(R, r=10, r_0=SUN_POSITION):
    return  2 * R * np.arccos((r_0**2 + R**2 -r**2) / (2*R*r_0)) * FerrierSource(R, r_0)

def source_near_sun(r):
    r_0 = SUN_POSITION
    return quad(source_near_sun_integrant,r_0-r,r_0+r,args=(r,r_0))[0]

thetaarray = np.load('thetafunction.npy')
def thetafunction(E, t):
    Dpcy1 = Dpcy
    H=HALO_HEIGHT
    h224Dt = (2*H)**2 / (4*D(E,n=1,D_0=Dpcy1,chi=1,delta=DIFFUSION_SPECTRUM)*t)
    xs = thetaarray[0]
    ys = thetaarray[1]
    return np.interp(h224Dt, xs, ys,0,1)

def Greensbounded(E, t, d2):
    Dpcy1 = Dpcy
    rd = 4 * D(E,n=1,D_0=Dpcy1,chi=1,delta=DIFFUSION_SPECTRUM) * t
    return (np.pi*rd)**(-3/2) * np.exp(-d2/rd) * thetafunction(E,t)

def integrant(t, R, theta, E):
    r_0 = SUN_POSITION
    d2 = (R*np.cos(theta)-r_0)**2 + (R*np.sin(theta))**2
    return R * Greensbounded(E, t, d2) * FerrierSource(R, r_0)

def integrant2(t, R, E):
    r_0 = SUN_POSITION
    Dpcy1 = Dpcy
    rd = 4 * D(E,n=1,D_0=Dpcy1,chi=1,delta=DIFFUSION_SPECTRUM) * t
    bes = 2 * R * r_0 / rd
    factor1 = (np.pi*rd)**(-3/2) * np.exp(-(R**2+r_0**2)/rd)
    return R * FerrierSource(R, r_0) * thetafunction(E,t) * factor1 * np.pi * 2*iv(0,bes)


def mean_bessel(E):
    low = 0 * 1e3**DIFFUSION_SPECTRUM / E**DIFFUSION_SPECTRUM
    high = 3e8 * 1e3**DIFFUSION_SPECTRUM / E**DIFFUSION_SPECTRUM
    return dblquad(integrant2,0,GALACTIC_RADIUS,low,high,args=(E,))[0]



# plt.loglog(Ea,escape_time(Ea)/YEAR_SECOND_CONVERSION)
# plt.savefig('test')


























print('runtime' , timeit.default_timer() - start)
print('runtime' , datetime.now() - startt)