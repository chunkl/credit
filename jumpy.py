import numpy as np
import matplotlib.pyplot as plt
import timeit
import multiprocessing
import matplotlib as mpl
#from joblib import Parallel, delayed
import multiprocessing
import matplotlib.colors as colors
#from scipy.integrate import tplquad, quad, quad_vec
from datetime import datetime
from cycler import cycler


#time the code
start = timeit.default_timer()
startt = datetime.now()
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
Ea = np.logspace(-3,np.log10(3e3),200)
NORMAL4 = np.loadtxt('normalisation4000')/4
R = 2e4
R1 = 1e4
N=500
distsq = np.linspace(0**2,2000**2,2*N)
dist0 = np.sqrt(distsq)
ages0 = np.linspace(0,2e5,N)
E = np.logspace(-3,np.log10(3e3),200)

def B(n=1):
    '''
    magnetic field strength in microGauss, n hydrogen per cm^3
    '''
    if n==1:
        return 3
    elif n<300:
        return 10
    else:
        return 10 * (n/300)**0.65

#diffusion coefficient
def D(E, n=1, D_0=DIFFUSION_COEFFICIENT, chi=1, delta=DIFFUSION_SPECTRUM):
    '''
    diffusion coefficient (cm2 s-1) on energy (TeV)
    '''
    return chi * D_0 * ((E*1000)/(B(n)/3))**delta

def newescape(E, t_sed=1.6e3, p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,  
              t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    yrtos = YEAR_SECOND_CONVERSION
    if t_max == 0:
        return yrtos * t_sed * np.exp((p_max/E - 1) / 1e4)
    elif t_max < t_sed:
        return yrtos * t_sed * (p_b1*np.log(p_max/E) + 1)
    elif t_b1<t_sed or p_b1>=p_max:
        return yrtos * t_sed
    elif t_b1==0 and t_b2==0:
        s1 = np.log(t_max/t_sed) / np.log(p_max/p_min)
        return yrtos * t_sed * (p_max/E) ** s1
    elif t_b2==0:
        s1 = np.log(t_b1/t_sed) / np.log(p_max/p_b1)
        s2 = np.log(t_max/t_b1) / np.log(p_b1/p_min)
        t1 = yrtos * t_b1 * (p_b1 / E[E<p_b1]) ** s2
        t2 = yrtos * t_sed * (p_max / E[E>p_b1]) ** s1
        return np.append(t1,t2)
    else:
        s1 = np.log(t_b1/t_sed) / np.log(p_max/p_b2)
        s2 = np.log(t_b2/t_b1) / np.log(p_b2/p_b1)
        s3 = np.log(t_max/t_b2) / np.log(p_b1/p_min)
        t1 = yrtos * t_b2 * (p_b1 / E[E<p_b1]) ** s3
        t2 = yrtos * t_b1 * (p_b2 / E[(E<p_b2) & (E>p_b1)]) ** s2
        t3 = yrtos * t_sed * (p_max / E[E>p_b2]) ** s1
        return np.append(np.append(t1,t2),t3)

def newR_dsq(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
             p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,
            t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,n,D_0,chi,delta) * (t_SNR[:,np.newaxis]*yrs - newescape(E,t_sed,p_max,t_max,p_min,t_b1,t_b2,p_b1,p_b2)) / pcm**2
    return result.T
plt.rc('font', size=22)
arrjump = np.zeros((2*N,N))
arre = np.zeros((2*N,N))
norm = NORMAL4 /4
for i,d2 in enumerate(distsq):
    rd1 = newR_dsq(E, ages0, p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3, t_b1=1e5-1, t_b2=0, p_b1=0.1, p_b2=0)
    rd = np.where(rd1>0, rd1, 1e-99)
    g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd)
    ratio = (g1.T / norm).T
    ratio1 = np.where(rd1>0, ratio, 0) 
    dif = np.diff(ratio1, axis=0)
    maxr = np.argmax(dif, axis=0)
    mean = np.max(dif,axis=0)
    arrjump[i] = np.array(mean)
    arre[i] = np.array(E[maxr])

#np.savetxt('jumps',arrjump)
#np.savetxt('energj',arre)

d,t = np.meshgrid(dist0,ages0)
norm1 = colors.LogNorm(1e-3,1e3)
plt.figure(0,figsize=(18,15))
plt.pcolormesh(d,t,arrjump.T,norm=norm1)
plt.colorbar()
cont0 = plt.contour(d,t,arrjump.T,levels=[0.001,0.01,0.1,1],colors='b',linewidths=1)
plt.clabel(cont0,fontsize='smaller',inline=0,colors='k',fmt='%1.3f')
plt.xlabel('diatance [pc]')
plt.ylabel('source age [years]')
plt.title('Jump Height')
plt.tight_layout()
plt.savefig('jumpsz.png',bbox_inches='tight')


plt.figure(1,figsize=(18,15))
plt.pcolormesh(d,t,arre.T,norm=norm1)
plt.colorbar()
cont0 = plt.contour(d,t,arre.T,levels=[0.001,0.01,0.1,1,10,100,1000],colors='b',linewidths=1)
plt.clabel(cont0,fontsize='smaller',inline=0,colors='k',fmt='%1.3f')
plt.xlabel('diatance [pc]')
plt.ylabel('source age [years]')

plt.title('Jump Rigidity')
plt.tight_layout()
plt.savefig('jumpez.png',bbox_inches='tight')
