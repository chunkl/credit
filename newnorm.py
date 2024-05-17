import numpy as np
import matplotlib.pyplot as plt
import timeit
import multiprocessing
import matplotlib as mpl
from joblib import Parallel, delayed
import multiprocessing


#time the code
start = timeit.default_timer()
c = 3e10 #cm/s
M_PI = 134.9768e-6 #TeV
M_PROTON = 938.27208816e-6 #TeV
T_TH = 2*M_PI + M_PI**2 / (2*M_PROTON)
YEAR_SECOND_CONVERSION = 31536000
CM_PC_CONVERSION = 3.0857e18
Earray = np.logspace(-2,5,1000)
#define constants
PROTON_INJECTION_SPECTRUM = 2
DIFFUSION_SPECTRUM = 0.6
ESCAPE_MOMENTUM_SPECTRUM = 2.5
DIFFUSION_SUPPRESSION_FACTOR = 0.1
DIFFUSION_COEFFICIENT = 3E26
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
ESCALING = 2


#diffusion coefficient
def D(E, D_0=3e26, delta=0.5):
    '''
    diffusion coefficient (cm2 s-1) on energy (TeV)
    '''
    return D_0 * ((E*1000))**delta



def R_dsq2(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 2 * D(E,D_0,delta) * (t_SNR*yrs) / pcm**2
    return result

def greensnppeak(E, t_SNR, d2,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    Rd2 = R_dsq2(E, t_SNR, D_0, delta)
    return  np.exp(-d2/Rd2)

def normdist(theta, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    rsq = np.linspace(0,20000**2,5000000)
    d0 = 8.3e3
    r = np.sqrt(rsq)
    E = np.logspace(-3,np.log10(3e3),200)
    d2 = r**2*np.sin(theta)**2 + (r*np.cos(theta)-d0)**2
    result = greensnppeak(E[:,np.newaxis],4e5,d2,D_0,delta)
    return np.mean(result,axis=1)
answer = np.zeros(200)
for i in range(1000):
    print(i)
    theta = 2*i/1000 * np.pi
    answer += normdist(theta)/1000
    if i==1 or i==0:
        print(answer)
    
np.savetxt('slice3',answer)
#answer = np.loadtxt('slice3')
H = 6
Ea = np.logspace(-3,np.log10(3e3),200)
#oneage = np.loadtxt('slice')
def tet(E):
    rd = R_dsq2(E, 4e5, DIFFUSION_COEFFICIENT, DIFFUSION_SPECTRUM)
    result = 1
    q = np.exp(-(2*H)**2/rd)
    for i in range(1,20):
        result *= (1-q**(2*i)) * (1-q**(2*i-1))**2
    return result
cor = answer * tet(Ea)
print(np.any(answer<0))
print(np.any(cor<=0))
def normage(distnorm, age1):
    Earr = Ea
    ts = np.linspace(1.7e3,1e9,100000000)
    result = np.zeros_like(Earr)
    for i, E in enumerate(Earr):
        print(i)
        rd = R_dsq2(E, ts, DIFFUSION_COEFFICIENT, DIFFUSION_SPECTRUM)
        changed = distnorm[i]**(age1/ts) * (np.pi*rd)**(-3/2)
        result[i] = np.sum(changed)
    return result * 3 #* 1e13

final = normage(cor, 4e5)
print(final)
np.savetxt('normfinal7',final)
#final = np.loadtxt('normfinal2')
plt.plot(Ea, final)
plt.xscale('log')
plt.yscale('log')
plt.savefig('normfinal.png')       
