import numpy as np
import matplotlib.pyplot as plt
import timeit
import multiprocessing
import matplotlib as mpl
from joblib import Parallel, delayed
import multiprocessing
from scipy.integrate import tplquad, quad, quad_vec


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
np.savetxt('test',Ea)

#diffusion coefficient
def D(E, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    '''
    diffusion coefficient (cm2 s-1) on energy (TeV)
    '''
    return D_0 * ((E*1000))**delta



def R_dsq2(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,D_0,delta) * (t_SNR*yrs) / pcm**2
    return result

def tet(E, t_SNR,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    rd = R_dsq2(E, t_SNR, D_0, delta)
    result = 1
    H=6000
    rd = np.where(rd>0,rd,-1)
    q = np.where(rd>0,np.exp(-(2*H)**2/rd),0)
    for i in range(1,50):
        result *= (1-q**(2*i)) * (1-q**(2*i-1))**2
    return result

def greensnppeak(E, t_SNR, d2,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
    Rd2 = R_dsq2(E, t_SNR, D_0, delta)
    Rd2 = np.where(Rd2>0,Rd2,1e-99)
    result = np.where(Rd2>1e-99,(Rd2)**(-3/2) * np.exp(-d2/Rd2) * tet(E, t_SNR, D_0, delta),(Rd2)**(-3/2) * tet(E, t_SNR, D_0, delta))
    return result

# def norm(E,D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM):
#     rsq = np.linspace(0,20000**2,40000)
#     d0 = 8.3e3
#     r0 = np.sqrt(rsq)
#     t = np.linspace(0,1e8,int(1e6))
#     theta0 = np.linspace(0,2*np.pi,500)
#     result = 0
#     for r in r0:
#         for theta in theta0:
#             print(r,theta)
#             d2 = r**2*np.sin(theta)**2 + (r*np.cos(theta)-d0)**2
#             between = greensnppeak(E,t,d2,D_0,delta)
#             result += np.mean(between)/500
#     np.savetxt('E'+str(E),np.array([result,0,0]))
#     return result



#oneage = np.loadtxt('slice')
flux = np.zeros(200)
Da = D(Ea)
R = 20000
H = 4000
d0 = 8.3e3

def Greens2(theta, r, t, E=Ea,
            D_0=Da, delta=DIFFUSION_SPECTRUM):
    d2 = r**2*np.sin(theta)**2 + (r*np.cos(theta)-d0)**2
    return greensnppeak(E,t,d2,D_0,delta)

def I1(r, t, E=Ea,
            D_0=Da, delta=DIFFUSION_SPECTRUM):
    return quad(Greens2,0,2*np.pi,limit=100,args=((r, t, E,D_0, delta)))[0]

def I2(t, E=Ea,
            D_0=Da, delta=DIFFUSION_SPECTRUM):
    return quad(I1,0,2e4,limit=int(1e4),args=(t, E,D_0, delta))[0]

def I3(E=Ea,
            D_0=Da, delta=DIFFUSION_SPECTRUM):
    return quad(I2,0,1e9,args=(E,D_0,delta))[0]

n = np.arange(100000)
def singleintegral(x):
    Ri = R
    r0 = d0
    Hi = H
    middle = np.sqrt(Ri**2-x**2) / np.sqrt((x-r0)**2 + Ri**2 - x**2 + (2*n*Hi)**2)
    return np.arctanh(middle)

def firstintegral(x):
    Ri = R
    r0 = d0
    middle = np.sqrt(Ri**2-x**2) / np.sqrt((x-r0)**2 + Ri**2 - x**2)
    return np.arctanh(middle)
# for i in range(200):
#     print(i)
#     def Greens2(theta, r, t, E=Ea[i],
#             D_0=Da[i], delta=DIFFUSION_SPECTRUM):
#         d2 = r**2*np.sin(theta)**2 + (r*np.cos(theta)-d0)**2
#         return greensnppeak(E,t,d2,D_0,delta)
#     flux[i] = tplquad(Greens2, 0, np.inf, 0, R, 0, 2*np.pi)[0]
#     print(flux[i])
#     print(timeit.default_timer() - start)
# #flux = tet(Ea,1e10)
# print(flux)
#np.savetxt('norm6001',flux)
# for i in range(200):
#     print(i)
#     flux[i] = I3(Ea[i], Da[i], DIFFUSION_SPECTRUM)
#     print(flux[i])
#     print(timeit.default_timer() - start)
conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
Dpcy = DIFFUSION_COEFFICIENT * conversion
#print(quad(singleintegral,0,R,points=(8.3e3,))[0])
interated = quad_vec(singleintegral,0,R,points=(8.3e3,))[0]
interated[0] = quad(firstintegral,0,R,points=(d0,))[0]
np.savetxt('normdist3000',interated)
normalisation = interated[0] + 2*np.sum((-1)**np.arange(100000)[1:]*interated[1:])
print(normalisation)
normalisation_energy = 1 / (np.pi * D(Ea,Dpcy, DIFFUSION_SPECTRUM)) * normalisation
corrected = normalisation_energy / (np.pi*R**2*100) * 3
#np.savetxt('normalisation4000',corrected)
#print(corrected*Ea**DIFFUSION_SPECTRUM)
plt.plot(Ea,corrected)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy/TeV')
plt.ylabel('CR density')
plt.savefig('normd.png')
#n0 = 42934.11116770028