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
C = 3e10 #cm/s
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
DIFFUSION_COEFFICIENT = 6.7825E27
MAXIMUM_PARTICLE_MOMENTUM = 3E3 #TeV
SEDOV_TIME = 1e3
SOURCE_LIFETIME = 1e5
MAXIMUM_RIGIDITY = 3e3
MINIMUM_RIGIDITY = 1e-3
GALACTIC_RADIUS = 15000
HALO_HEIGHT = 4000
SUN_POSITION = 8300


approx_AMS = np.sqrt(1.16)*1.10655**np.arange(72)
approx_DAMPE = 1983*1.58308693**np.arange(9)
AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY = np.concatenate((approx_AMS, approx_DAMPE))/1000

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
    h224Dt = (2*H)**2 / (4*D(E,D_0=Dpcy1,delta=DIFFUSION_SPECTRUM)*t)
    xs = thetaarray[0]
    ys = thetaarray[1]
    return np.interp(h224Dt, xs, ys,0,1)

def Greensbounded(E, t, d2):
    Dpcy1 = Dpcy
    rd = 4 * D(E,D_0=Dpcy1,delta=DIFFUSION_SPECTRUM) * t
    return (np.pi*rd)**(-3/2) * np.exp(-d2/rd) * thetafunction(E,t)

def integrant(t, R, theta, E):
    r_0 = SUN_POSITION
    d2 = (R*np.cos(theta)-r_0)**2 + (R*np.sin(theta))**2
    return R * Greensbounded(E, t, d2) * FerrierSource(R, r_0)

def integrant2(t, R, E):
    r_0 = SUN_POSITION
    Dpcy1 = Dpcy
    rd = 4 * D(E,D_0=Dpcy1,delta=DIFFUSION_SPECTRUM) * t
    bes = 2 * R * r_0 / rd
    factor1 = (np.pi*rd)**(-3/2) * np.exp(-(R**2+r_0**2)/rd)
    return R * FerrierSource(R, r_0) * thetafunction(E,t) * factor1 * np.pi * 2*iv(0,bes)

def mean_bessel(E):
    low = 0 * 1e3**DIFFUSION_SPECTRUM / E**DIFFUSION_SPECTRUM
    high = 3e8 * 1e3**DIFFUSION_SPECTRUM / E**DIFFUSION_SPECTRUM
    return dblquad(integrant2,0,GALACTIC_RADIUS,low,high,args=(E,))[0]

def diffusion_length(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,D_0,delta) * (t_SNR[:,np.newaxis]*yrs - escape_time(E,p_break,t_life,p_max,p_min,t_break)) / pcm**2
    return result

def diffusion_length1(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,D_0,delta) * (t_SNR*yrs - escape_time(E,p_break,t_life,p_max,p_min,t_break)) / pcm**2
    return result

def causal_length(E, t_SNR,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    c = C
    result = c * (t_SNR[:,np.newaxis]*yrs - escape_time(E,p_break,t_life,p_max,p_min,t_break)) / pcm
    return result

def causal_length1(E, t_SNR,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    c = C
    result = c * (t_SNR*yrs - escape_time(E,p_break,t_life,p_max,p_min,t_break)) / pcm
    return result

def thetafunction_credit(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY,
                     p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001, H=HALO_HEIGHT):
    rd = diffusion_length(E,t_SNR,D_0,delta,p_break,t_life,p_max,p_min,t_break)
    h224Dt = (2*H)**2 / rd
    xs = thetaarray[0]
    ys = thetaarray[1]
    return np.interp(h224Dt, xs, ys,0,1)

def greens_credit(E, t_SNR, d, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY,
                     p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001, H=HALO_HEIGHT):
    rd2 = diffusion_length1(E,t_SNR,D_0,delta,p_break,t_life,p_max,p_min,t_break)
    rd = np.where(rd2>0, rd2, 1e-99)
    h224Dt = (2*H)**2 / rd
    xs = thetaarray[0]
    ys = thetaarray[1]
    bound =  np.interp(h224Dt, xs, ys,0,1)
    ct2 = causal_length1(E, t_SNR, p_break, t_life, p_max, p_min,t_break)
    g2 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd) * bound
    g21 = np.where(d<ct2, g2, 0)
    G2 = np.where(rd2>0, g21, 0)
    return G2

def to_integrate(t_SNR,E, d, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM,
                     p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY,
                     p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001, H=HALO_HEIGHT):
    return greens_credit(E, t_SNR, d, D_0, delta,p_break, t_life, p_max,p_min,t_break, H)
# a=np.zeros((1000,9))
# for i in range(1000):
#     for j in range(9):
#         a[i,j] = np.array(quad(to_integrate,0,SOURCE_LIFETIME,args=(approx_DAMPE[j]/1000,4*i))[1])
e = np.load('green_sum_integrated_error.npy')
g = np.load('green_sum_integrated.npy')
plt.plot(4*np.arange(1000),e/g)
plt.yscale('log')
plt.xscale('log')
plt.savefig('green_sum_integrated_error')

def debugger(d=300, Nd=1500, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME,
            p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001,
            alpha=PROTON_INJECTION_SPECTRUM):
    Nd = int(Nd)
    t = np.linspace(0,t_life,Nd)
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))
    Elow = np.ones(len(E)-1)
    Ehigh = E[1:] - E[:-1] + 1
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation = mean_bessel(0.001) * np.pi * D(0.001,Dpcy,delta)
    norm = normalisation / (np.pi*D(ED)) * ED**(-alpha)
    normall = normalisation / (np.pi*D(E)) * E**(-alpha)
    slopep = np.log(normall[1:]/normall[:-1]) / (np.log(Ehigh/Elow))
    normi1 = normall[:-1]/(slopep+1)*(Ehigh**(slopep+1)-Elow**(slopep+1))
    normi = np.concatenate((np.sum(normi1[:720].reshape(72,10),axis=1) ,
                             np.sum(normi1[721:].reshape(9,10),axis=1))) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    rd1 = diffusion_length(E, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
    rd = np.where(rd1>0, rd1, 1e-99)
    ct1 = causal_length(E, t, p_break, t_life, p_max, p_min,t_break)
    g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd) * E**(-alpha)
    g11 = np.where(d<ct1, g1, 0)
    G = np.where(rd1>0, g11, 0) + normall
    slope = np.log(G[:,1:]/G[:,:-1]) / (np.log(Ehigh/Elow))
    gi1 = np.where(slope==-1, G[:,:-1]*np.log(Ehigh/Elow) , G[:,:-1]/(slope+1)*(Ehigh**(slope+1)-Elow**(slope+1)))
    gi = np.concatenate((np.sum(gi1[:,:720].reshape(Nd,72,10),axis=2) ,
                             np.sum(gi1[:,721:].reshape(Nd,9,10),axis=2)), axis=1) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    rd2 = diffusion_length(ED, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
    rd3 = np.where(rd2>0, rd2, 1e-99)
    ct2 = causal_length(ED, t, p_break, t_life, p_max, p_min,t_break)
    g2 = (np.pi*rd3)**(-3/2) * np.exp(-d**2/rd3) * ED**(-alpha)
    g21 = np.where(d<ct2, g2, 0)
    G2 = np.where(rd2>0, g21, 0)
    fluxi = np.sum(gi-normi,axis=0)
    fluxp = np.sum(G2,axis=0)
    return fluxi

def debugger_point(d=300, Nd=1500, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME,
            p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001,
            alpha=PROTON_INJECTION_SPECTRUM,H=HALO_HEIGHT):
    Nd = int(Nd)
    t = np.linspace(0,t_life,Nd)
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))[-9:]
    rd2 = diffusion_length(ED, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
    rd3 = np.where(rd2>0, rd2, 1e-99)
    ct2 = causal_length(ED, t, p_break, t_life, p_max, p_min,t_break)
    g2 = (np.pi*rd3)**(-3/2) * np.exp(-d**2/rd3) *thetafunction_credit(ED,t,D_0,delta,p_break,t_life,p_max,p_min,t_break,H)
    g21 = np.where(d<ct2, g2, 0)
    G2 = np.where(rd2>0, g21, 0)
    fluxp = np.sum(G2,axis=0)
    return fluxp

# test = np.zeros((1000,9))
# for i in range(1000):
#     test[i] = debugger_point(4*i+4)
# np.save('greens_sum_point_boundedup',test)

def debugger2(num_source=1, d=300, Nd=1500, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME,
            p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001,
            alpha=PROTON_INJECTION_SPECTRUM, H=HALO_HEIGHT):
    Nd = int(Nd)
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))
    Elow = np.ones(len(E)-1)
    Ehigh = E[1:] - E[:-1] + 1
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation = mean_bessel(0.001) * np.pi * D(0.001,Dpcy,delta)
    norm = normalisation / (np.pi*D(ED,Dpcy,delta)) * ED**(-alpha)
    normall = normalisation / (np.pi*D(E,Dpcy,delta)) * E**(-alpha)
    slopep = np.log(normall[1:]/normall[:-1]) / (np.log(Ehigh/Elow))
    normi1 = normall[:-1]/(slopep+1)*(Ehigh**(slopep+1)-Elow**(slopep+1))
    normi = np.concatenate((np.sum(normi1[:720].reshape(72,10),axis=1) ,
                             np.sum(normi1[721:].reshape(9,10),axis=1))) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat1 = np.concatenate((1e-4 * (Edge[1:72]/1e-3) ** power / 1.5 , 1e-4 * (Edge[74:-1]/1e-3) ** power / 1.5))
    jumplimstat = np.where(np.delete(ED,(71,-1))<p_break/2,1e10,jumplimstat1)
    sigmas = 0.1 + 0.2*np.arange(26)
    resultint = np.zeros_like(sigmas)
    resultpoint = np.zeros_like(sigmas)
    for j in range(100):
        Gint = np.zeros((Nd,len(ED)))
        Gint1 = np.zeros((Nd,len(E)))
        Gpoint = np.zeros((Nd,len(ED)))
        for k in range(num_source):
            t = np.random.uniform(0,10*num_source*t_life,Nd)
            rd1 = diffusion_length(E, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
            rd = np.where(rd1>0, rd1, 1e-99)
            ct1 = causal_length(E, t, p_break, t_life, p_max, p_min,t_break)
            g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd) * E**(-alpha) * thetafunction_credit(E,t,D_0,delta,p_break,t_life,p_max,p_min,t_break,H)
            g11 = np.where(d<ct1, g1, 0)
            G = np.where(rd1>0, g11, 0) 
            rd2 = diffusion_length(ED, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
            rd3 = np.where(rd2>0, rd2, 1e-99)
            ct2 = causal_length(ED, t, p_break, t_life, p_max, p_min,t_break)
            g2 = (np.pi*rd3)**(-3/2) * np.exp(-d**2/rd3) * ED**(-alpha) * thetafunction_credit(ED,t,D_0,delta,p_break,t_life,p_max,p_min,t_break,H)
            g21 = np.where(d<ct2, g2, 0)
            G2 = np.where(rd2>0, g21, 0)
            Gint1 += G
            Gpoint +=G2
        Gint = Gint1 + normall
        slope = np.log(Gint[:,1:]/Gint[:,:-1]) / (np.log(Ehigh/Elow))
        gi1 = np.where(slope==-1, Gint[:,:-1]*np.log(Ehigh/Elow) , Gint[:,:-1]/(slope+1)*(Ehigh**(slope+1)-Elow**(slope+1)))
        gi = np.concatenate((np.sum(gi1[:,:720].reshape(Nd,72,10),axis=2) ,
                             np.sum(gi1[:,721:].reshape(Nd,9,10),axis=2)), axis=1) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
        ratio1 = (gi-normi)/normi
        ratio2 = Gpoint/norm
        dif1 = np.diff(ratio1,axis=1)
        dif2 = np.diff(ratio2,axis=1)
        for i,sigma in enumerate(sigmas):
            resultint[i] += np.sum(np.any(np.delete(dif1,71,axis=1)>sigma*jumplimstat, axis=1))
            resultpoint[i] += np.sum(np.any(np.delete(dif2,71,axis=1)>sigma*jumplimstat, axis=1))
    return np.append(resultint/(100*Nd), resultpoint/(100*Nd))

def debugger3(num_source=1, Nd=1500, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME,
            p_break=10, t_life=SOURCE_LIFETIME, p_max=MAXIMUM_RIGIDITY, p_min=MINIMUM_RIGIDITY,t_break=SOURCE_LIFETIME-0.001,
            alpha=PROTON_INJECTION_SPECTRUM, H=HALO_HEIGHT):
    Nd = int(Nd)
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))
    Elow = np.ones(len(E)-1)
    Ehigh = E[1:] - E[:-1] + 1
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation = mean_bessel(0.001) * np.pi * D(0.001,Dpcy,delta)
    norm = normalisation / (np.pi*D(ED,Dpcy,delta)) * ED**(-alpha)
    normall = normalisation / (np.pi*D(E,Dpcy,delta)) * E**(-alpha)
    slopep = np.log(normall[1:]/normall[:-1]) / (np.log(Ehigh/Elow))
    normi1 = normall[:-1]/(slopep+1)*(Ehigh**(slopep+1)-Elow**(slopep+1))
    normi = np.concatenate((np.sum(normi1[:720].reshape(72,10),axis=1) ,
                             np.sum(normi1[721:].reshape(9,10),axis=1))) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat1 = np.concatenate((1e-4 * (Edge[1:72]/1e-3) ** power / 1.5 , 1e-4 * (Edge[74:-1]/1e-3) ** power / 1.5))
    jumplimstat = np.where(np.delete(ED,(71,-1))<p_break/2,1e10,jumplimstat1)
    sigmas = 0.1 + 0.2*np.arange(26)
    resultint = np.zeros_like(sigmas)
    resultpoint = np.zeros_like(sigmas)
    for j in range(150):
        print(j)
        Gint = np.zeros((Nd,len(ED)))
        Gint1 = np.zeros((Nd,len(E)))
        Gpoint = np.zeros((Nd,len(ED)))
        for k in range(num_source):
            t = np.random.uniform(0,10*num_source*t_life,Nd)
            d2 = np.random.uniform(0,2000**2,Nd)[:,np.newaxis]
            rd1 = diffusion_length(E, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
            rd = np.where(rd1>0, rd1, 1e-99)
            ct1 = causal_length(E, t, p_break, t_life, p_max, p_min,t_break)
            g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd) * E**(-alpha) * thetafunction_credit(E,t,D_0,delta,p_break,t_life,p_max,p_min,t_break,H)
            g11 = np.where(np.sqrt(d2)<ct1, g1, 0)
            G = np.where(rd1>0, g11, 0) 
            rd2 = diffusion_length(ED, t, D_0, delta, p_break, t_life, p_max, p_min,t_break)
            rd3 = np.where(rd2>0, rd2, 1e-99)
            ct2 = causal_length(ED, t, p_break, t_life, p_max, p_min,t_break)
            g2 = (np.pi*rd3)**(-3/2) * np.exp(-d2/rd3) * ED**(-alpha) * thetafunction_credit(ED,t,D_0,delta,p_break,t_life,p_max,p_min,t_break,H)
            g21 = np.where(np.sqrt(d2)<ct2, g2, 0)
            G2 = np.where(rd2>0, g21, 0)
            Gint1 += G
            Gpoint +=G2
        Gint = Gint1 + normall
        slope = np.log(Gint[:,1:]/Gint[:,:-1]) / (np.log(Ehigh/Elow))
        gi1 = np.where(slope==-1, Gint[:,:-1]*np.log(Ehigh/Elow) , Gint[:,:-1]/(slope+1)*(Ehigh**(slope+1)-Elow**(slope+1)))
        gi = np.concatenate((np.sum(gi1[:,:720].reshape(Nd,72,10),axis=2) ,
                             np.sum(gi1[:,721:].reshape(Nd,9,10),axis=2)), axis=1) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
        ratio1 = (gi-normi)/normi
        ratio2 = Gpoint/norm
        dif1 = np.diff(ratio1,axis=1)
        dif2 = np.diff(ratio2,axis=1)
        for i,sigma in enumerate(sigmas):
            resultint[i] += np.sum(np.any(np.delete(dif1,71,axis=1)>sigma*jumplimstat, axis=1))
            resultpoint[i] += np.sum(np.any(np.delete(dif2,71,axis=1)>sigma*jumplimstat, axis=1))
    return np.append(resultint/(150*Nd), resultpoint/(150*Nd))

#test = np.zeros((15,52))
# for i in range(15):
#     start2 = timeit.default_timer()
#     test[i] = debugger2(i+1,d=300,Nd=1e3)
#     print(i, timeit.default_timer() - start2)
# test = debugger3(63,1e3)
# print(test)
# np.save('numb_sources_63_range_10xtime_1.5e5realizationalt',test)

# mymean = np.array([mean_bessel(E) for E in AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY])
#Antonsmean = np.load('mean_Anton.npy')*(1000*AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)**PROTON_INJECTION_SPECTRUM

# plt.loglog(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,mymean)
# plt.loglog(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,Antonsmean)
# plt.savefig('test')


























print('runtime' , timeit.default_timer() - start)
print('runtime' , datetime.now() - startt)