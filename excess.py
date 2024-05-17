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
LY_PC_CONVERSION = 3.26156378
T_TH = 2*M_PI + M_PI**2 / (2*M_PROTON)
YEAR_SECOND_CONVERSION = 31557600
CM_PC_CONVERSION = 3.0857e18
#define constants
GALACTIC_RADIUS = 15000 #pc
HALO_HEIGHT = 4000 #pc
SUN_POSITION = 8300 #pc
SOURCE_RATE = 3 #/century
SOURCE_LIFETIME = 1e5 #yr
SEDOV_TIME = 1e3 #yr
MIN_RIGIDITY = 1e-3 #TV
MAX_RIGIDITY = 3e3 #TV
PROTON_INJECTION_SPECTRUM = 2.2
DIFFUSION_SPECTRUM = 0.6
ESCAPE_MOMENTUM_SPECTRUM = 2.5
DIFFUSION_SUPPRESSION_FACTOR = 0.1
DIFFUSION_COEFFICIENT = 6.7825E27 # cm2 s-1
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


approx_AMS = np.sqrt(1.16)*1.10655**np.arange(72)
approx_DAMPE = 1983*1.58308693**np.arange(9)
AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY = np.concatenate((approx_AMS, approx_DAMPE))/1000
meanA = np.load('mean_Anton.npy')*(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY*1000)**PROTON_INJECTION_SPECTRUM

Amean = interpolate.interp1d(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY, meanA, fill_value='extrapolate')


def radial(R, r_0=SUN_POSITION):
    result = 4.745 * np.exp(-(R-r_0)/4500)
    if R>3700:
        result += 26.679 * np.exp(-(R**2 - r_0**2)/(6800**2))
    else:
        result += 26.679 * np.exp(-(3700**2 - r_0**2)/(6800**2)) * np.exp(-((R - 3700)/2100)**2)
    return result

sourcechange = quad(lambda R: 2*np.pi*R*radial(R), 0, GALACTIC_RADIUS)[0]

def FerrierSource(R, r_0=SUN_POSITION):
    '''
    in source per pc^2 year integrated to give 3 per cnetury in galaxy
    '''
    normalisation = sourcechange
    result = 4.745 * np.exp(-(R-r_0)/4500)
    if R>3700:
        result += 26.679 * np.exp(-(R**2 - r_0**2)/(6800**2))
    else:
        result += 26.679 * np.exp(-(3700**2 - r_0**2)/(6800**2)) * np.exp(-((R - 3700)/2100)**2)
    return result / normalisation *3 /100

#def greensfactor(R,theta,r_0=SUN_POSITION,H=HALO_HEIGHT,n=0):
    return R / np.sqrt((R*np.cos(theta)-r_0)**2 + R**2*np.sin(theta)**2 + (2*n*H)**2)


#def integrant(R, theta, r_0=SUN_POSITION, H=HALO_HEIGHT, n=0):
    return greensfactor(R,theta,r_0,H,n)*FerrierSource(R,r_0)


#galacticintegral = np.array([dblquad(integrant,0,2*np.pi,0,GALACTIC_RADIUS, args=(SUN_POSITION, HALO_HEIGHT, n))[0] for n in range(1000001)])
#np.savetxt('galacticintegral',galacticintegral)
galacticintegral = np.loadtxt('galacticintegral')

#galacticintegral2 = np.array([dblquad(integrant,0,2*np.pi,0,GALACTIC_RADIUS, args=(SUN_POSITION, HALO_HEIGHT, n))[0] for n in range(20)])

def sunintegrant(R,r =10, r_0=8300):
    return 2 * R * np.arccos((r_0**2 + R**2 -r**2) / (2*R*r_0)) * FerrierSource(R, r_0)

def sunintegral(r, r_0=8300):
    return quad(sunintegrant,r_0-r,r_0+r,args=(r,r_0))[0]


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

#magnetic field
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

#cross section
def sigma_kafexhiu(E):
    '''
    total inelastic cross section of pp to gamma from kafexhiu, energy in TeV
    '''
    T_th = T_TH
    mp = M_PROTON
    T = E - mp
    L = np.log(T/T_th)
    suppression = (1 - (T_th/T)**1.9)**3
    result = np.where(suppression>=0, (30.7 - 0.96*L +0.18*L**2) * suppression, 0)
    mbtocm = 1e27
    return result  / mbtocm

#proton lifetime against interaction
def tau_pp(E, n=1, kappa=0.45):
    '''
    lifetime in s
    '''
    return 1 / (n * C * kappa * sigma_kafexhiu(E))

#escape time
def t_esc(E, t_sed=1.6e3, p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=2.5):
    '''
    time of escape of CR protons depending on momentum (energy in TeV via Einstein relation)
    '''
    yrtos = YEAR_SECOND_CONVERSION
    return t_sed * yrtos * (E/p_m)**(-1/beta)


def R_dsq1(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1, p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,n,D_0,chi,delta) * (t_SNR*yrs - t_esc(E, t_sed, p_m, beta)) / pcm**2
    return result

def R_dsq2(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, n=1, chi=1):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,n,D_0,chi,delta) * (t_SNR*yrs) / pcm**2
    return result

def greensnppeak(E, t_SNR, d,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, n=1, chi=1):
    Rd2 = R_dsq2(E, t_SNR, D_0, delta, n, chi)
    return (np.pi*Rd2)**(-3/2) * np.exp(-d**2/Rd2)

conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
Dpcy = DIFFUSION_COEFFICIENT * conversion
normalisation = galacticintegral[0] + 2*np.sum((-1)**np.arange(1000001)[1:]*galacticintegral[1:])
#normalE = 1 / (4 * np.pi * D(Ea,Dpcy, DIFFUSION_SPECTRUM)) * normalisation

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

# 0.000003457905225482213

def individual_trapezoidal_rule_log(start_x, stop_x, start_y, stop_y):
        stop_x=stop_x-start_x+1
        start_x=np.ones_like(start_x)
        b = np.log10(stop_y/start_y)/np.log10(stop_x/start_x)
        a = start_y
        return a/(b+1) * (stop_x**(b+1)-start_x**(b+1))

def excess2(t_SNR, d,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    E = np.logspace(-4,np.log10(p_m),50000)
    rd1 = R_dsq1(E,t_SNR,D_0,delta,t_sed,n,chi,p_m,beta)
    rd = np.where(rd1>0, rd1, 0)
    g1 = rd**(-3/2) * np.exp(-d**2/rd)
    g2 = greensnppeak(E, t_SNR, d, D_0, delta, n, chi)
    ratio = g1 / g2
    ratio1 = np.where(rd1>0, ratio, 1)
    highest = np.max(ratio1)
    emax = np.argmax(ratio1)
    hm = np.where(ratio>(highest/2+0.5))[0]
    if len(hm) == 0:
        fwhm = 0
    else:
        hm1 = hm[-1]
        hm2 = hm[0]
        if hm1>emax:
            fwhm = E[hm1] - E[hm2]
        else:
            fwhm = 0
    return highest, fwhm, E[emax]

'''distances = np.logspace(0,3,5000)
ages = np.logspace(3,6,1000)
result = np.zeros((5000,1000,3))
for i,d in enumerate(distances):
    print(i)
    for j,t in enumerate(ages):
       result[i,j] = excess2(t,d)'''

def excessbar(t_SNR, d, Emid=1, barsize=25,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    '''
    Emid is the quoted energy, in TeV
    barsize in percentage
    '''
    percent = barsize/100
    E = np.logspace(np.log10(Emid*(1-percent)),np.log10(Emid*(1+percent)),50)
    rd1 = R_dsq1(E,t_SNR,D_0,delta,t_sed,n,chi,p_m,beta)
    rd = np.where(rd1>0, rd1, 0)
    g1 = rd**(-3/2) * np.exp(-d**2/rd)
    g2 = greensnppeak(E, t_SNR, d, D_0, delta, n, chi)
    ratio = g1 / g2
    ratio1 = np.where(rd1>0, ratio, 1)
    return np.mean(ratio1)

def excessconstantage(t_SNR=8e3, Emid=50, yerr=0.75, barsize=50,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    '''
    takes age of SNR in year,
    returns probability that excess more than errorbar of experiment,
    assuming a log distribution of distance form SNR
    '''
    Nd = 2000
    distsq = np.linspace(1,300**2,Nd)
    dist = np.sqrt(distsq)
    result = np.zeros(Nd)
    for i,d in enumerate(dist):
        excesss = excessbar(t_SNR, d, Emid, barsize, D_0, delta, t_sed, n, chi, p_m, beta) - 1
        if excesss > yerr:
            result[i] = 1
        else:
            continue
    return np.sum(result)/Nd

def excessconstantdist(d=57, Emid=1, yerr=0.75, barsize=25,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    '''
    takes distance in pc,
    returns probability that excess more than errorbar of experiment,
    assuming a log distribution of SNR age
    '''
    Nd = 3000
    age = np.linspace(1e3,1e6,Nd)
    result = np.zeros(Nd)
    for i,t in enumerate(age):
        excesss = excessbar(t, d, Emid, barsize, D_0, delta, t_sed, n, chi, p_m, beta) - 1
        if excesss > yerr:
            result[i] = 1
        else:
            continue
    return np.sum(result)/Nd

def excessxy(Emid=1, flux=0.1, barsize=25, yerr=1,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=1000
    distsq = np.linspace(1,300**2,Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(1e3,3e5,Nd)
    ages = np.array(np.meshgrid(ages0,dist0))[0].flatten()
    dist = np.array(np.meshgrid(ages0,dist0))[1].flatten()
    percent = barsize/100
    E = np.logspace(np.log10(Emid/(1+percent)),np.log10(Emid*(1+percent)),50)
    rd1 = R_dsq1(E[:,np.newaxis],ages,D_0,delta,t_sed,n,chi,p_m,beta)
    rd = np.where(rd1>0, rd1, 0)
    g1 = rd**(-3/2) * np.exp(-dist**2/rd)
    g2 = greensnppeak(E[:,np.newaxis], ages, dist, D_0, delta, n, chi)
    ratio = g1 / g2
    ratio1 = np.where(rd1>0, ratio, 1)
    cut = np.logical_and(np.any(g1>np.max(flux*(1-yerr),0),axis=0) , np.any(g1<flux*(1+yerr),axis=0))
    mean = np.mean(ratio1[:,cut],axis=0)
    peak = mean[mean>8]
    if np.size(mean) == 0:
        answer = 0
    else:
        answer = np.size(peak) / np.size(mean)
    return answer, len(mean), len(peak)

def excessfit(Emid=1, flux=0.1, barsize=25, yerr=1,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=1000
    distsq = np.linspace(1,300**2,Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(1e3,3e5,Nd)
    ages = np.array(np.meshgrid(ages0,dist0))[0].flatten()
    dist = np.array(np.meshgrid(ages0,dist0))[1].flatten()
    percent = barsize/100
    E = np.logspace(np.log10(Emid/(1+percent)),np.log10(Emid*(1+percent)),50)
    rd1 = R_dsq1(E[:,np.newaxis],ages,D_0,delta,t_sed,n,chi,p_m,beta)
    rd = np.where(rd1>0, rd1, 0)
    g1 = rd**(-3/2) * np.exp(-dist**2/rd)
    g2 = greensnppeak(E[:,np.newaxis], ages, dist, D_0, delta, n, chi)
    fitpeak = np.logical_and(np.any(g1>np.max(flux*(1-yerr),0),axis=0) , np.any(g1<flux*(1+yerr),axis=0))
    fitnone = np.logical_and(np.any(g2>np.max(flux*(1-yerr),0),axis=0) , np.any(g2<flux*(1+yerr),axis=0))
    ratio = g1 / g2
    ratio1 = np.where(rd1>0, ratio, 1) - 1
    mean = np.mean(ratio1,axis=0)
    return np.sum(fitpeak[mean>=yerr]), np.sum(fitnone) + np.sum(fitpeak[mean<yerr])

def excessall(Emid=1, barsize=25, excesslim=7, life = 1e6,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=1000
    distsq = np.linspace(1,300**2,Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(t_sed,3e5,Nd)
    ages = np.array(np.meshgrid(ages0,dist0))[0].flatten()
    dist = np.array(np.meshgrid(ages0,dist0))[1].flatten()
    percent = barsize/100
    E = np.logspace(np.log10(Emid/(1+percent)),np.log10(Emid*(1+percent)),50)
    rd1 = R_dsq1(E[:,np.newaxis],ages,D_0,delta,t_sed,n,chi,p_m,beta)
    rd = np.where(rd1>0, rd1, 1e-99)
    g2 = greensnppeak(E[:,np.newaxis], ages, dist, D_0, delta, n, chi)
    g1 = rd**(-3/2) * np.exp(-dist**2/rd)
    enoughflux = np.logical_and(np.all(g2>1e-51, axis=0) , np.any(g1>1e-50, axis=0))
    ratio = g1 / np.where(g2>1e-51, g2, 1)
    ratio1 = np.where(rd1>0, ratio, 1) - 1
    mean = np.mean(ratio1,axis=0)[enoughflux]
    peak = mean[mean>excesslim]
    return len(mean), len(peak)

def releseslope(lifetime, t_sed=SEDOV_SNR_II, p_m=MAXIMUM_PARTICLE_MOMENTUM, p_min=1e-3):
    x = np.log(lifetime) - np.log(t_sed)
    y = np.log(p_m) - np.log(p_min)
    return y / x

def excessingle(excesslim=7, life = 1e6, maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=3000
    distsq = np.linspace(1,maxdist**2,Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(t_sed,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    accepted = 0
    peaks = 0
    for d in dist0:
        rd1 = R_dsq1(E[:,np.newaxis],ages0,D_0,delta,t_sed,n,chi,p_m,beta)
        rd = np.where(rd1>0, rd1, 1e-99)
        g2 = greensnppeak(E[:,np.newaxis], ages0, d, D_0, delta, n, chi)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
        enoughflux = np.logical_and(np.all(g2>1e-51, axis=0) , np.any(g1>1e-50, axis=0))
        ratio = g1 / np.where(g2>1e-51, g2, 1)
        ratio1 = np.where(rd1>0, ratio, 1) - 1
        mean = np.max(ratio1,axis=0)[enoughflux]
        peak = mean[mean>excesslim]
        accepted += len(mean)
        peaks += len(peak)
    return peaks / accepted

def excesscompare(jumplim=0.05, maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(t_sed,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    norm = 1
    accepted = 0
    peaks = 0
    peaks1 = 0
    peaks2 = 0
    peaks3 = 0
    peaks4 = 0
    peaks5 = 0
    for d in dist0:
        rd1 = R_dsq1(E[:,np.newaxis],ages0,D_0,delta,t_sed,n,chi,p_m,beta)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
        ratio = (g1.T / norm).T
        ratio1 = np.where(rd1>0, ratio, 1) - 1
        dif = np.diff(ratio1, axis=0)
        mean = np.max(dif[:40],axis=0)
        peak = mean[mean>jumplim]
        mean1 = np.max(dif[40:80],axis=0)
        peak1 = mean1[mean1>jumplim]
        mean2 = np.max(dif[80:120],axis=0)
        peak2 = mean2[mean2>jumplim]
        mean3 = np.max(dif[120:160],axis=0)
        peak3 = mean3[mean3>jumplim]
        mean4 = np.max(dif[160:],axis=0)
        peak4 = mean4[mean4>jumplim]
        mean5 = np.max(dif,axis=0)
        peak5 = mean5[mean5>jumplim]
        accepted += len(mean5)
        peaks += len(peak)
        peaks1 += len(peak1)
        peaks2 += len(peak2)
        peaks3 += len(peak3)
        peaks4 += len(peak4)
        peaks5 += len(peak5)
    return np.array([accepted,peaks5,peaks,peaks1,peaks2,peaks3,peaks4])

def excesscompare2(jumplim=0.05, maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(t_sed,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    norm = 1
    accepted = 0
    peaks = 0
    peaks1 = 0
    peaks2 = 0
    peaks3 = 0
    peaks4 = 0
    peaks5 = 0
    for d in dist0:
        rd1 = R_dsq1(E[:,np.newaxis],ages0,D_0,delta,t_sed,n,chi,p_m,beta)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
        ratio = (g1.T / norm).T
        ratio1 = np.where(rd1>0, ratio, 1) - 1
        dif = np.gradient(ratio1, axis=0)
        maxr = np.argmax(dif, axis=0)
        mean = np.max(dif,axis=0)
        peak = np.logical_and(mean>jumplim, True)
        peak1 = np.logical_and(mean>jumplim, maxr<40)
        peak2 = np.logical_and.reduce([mean>jumplim, maxr<80, maxr>=40])
        peak3 = np.logical_and.reduce([mean>jumplim, maxr<120, maxr>=80])
        peak4 = np.logical_and.reduce([mean>jumplim, maxr<160, maxr>=120])
        peak5 = np.logical_and(mean>jumplim, maxr>=160)
        accepted += len(mean)
        peaks += np.sum(peak)
        peaks1 += np.sum(peak1)
        peaks2 += np.sum(peak2)
        peaks3 += np.sum(peak3)
        peaks4 += np.sum(peak4)
        peaks5 += np.sum(peak5)
    return np.array([accepted,peaks,peaks1,peaks2,peaks3,peaks4,peaks5])

def compare2d(maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_m=MAXIMUM_PARTICLE_MOMENTUM, beta=ESCAPE_MOMENTUM_SPECTRUM):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(t_sed,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    norm = 1
    Nr = 45
    result = np.zeros(Nr)
    Ju = np.linspace(0,0.3,Nr)
    for i,jumplim in enumerate(Ju):
        print(i)
        accepted = 0
        peaks = 0
        for d in dist0:
            rd1 = R_dsq1(E[:,np.newaxis],ages0,D_0,delta,t_sed,n,chi,p_m,beta)
            rd = np.where(rd1>0, rd1, 1e-99)
            g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
            ratio = (g1.T / norm).T
            ratio1 = np.where(rd1>0, ratio, 1) - 1
            dif = np.gradient(ratio1, axis=0)
            mean = np.max(dif,axis=0)
            peak = np.logical_and(mean>jumplim, True)
            accepted += len(mean)
            peaks += np.sum(peak)
        result[i] = peaks / accepted
    return result

def newescape(E, t_sed=SEDOV_TIME, p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
              t_b1=SOURCE_LIFETIME-1, t_b2=0, p_b1=10, p_b2=0):
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
        t2 = yrtos * t_sed * (p_max / E[E>=p_b1]) ** s1
        return np.append(t1,t2)
    else:
        s1 = np.log(t_b1/t_sed) / np.log(p_max/p_b2)
        s2 = np.log(t_b2/t_b1) / np.log(p_b2/p_b1)
        s3 = np.log(t_max/t_b2) / np.log(p_b1/p_min)
        t1 = yrtos * t_b2 * (p_b1 / E[E<p_b1]) ** s3
        t2 = yrtos * t_b1 * (p_b2 / E[(E<p_b2) & (E>=p_b1)]) ** s2
        t3 = yrtos * t_sed * (p_max / E[E>=p_b2]) ** s1
        return np.append(np.append(t1,t2),t3)

def newR_dsq(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
             p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,
            t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,n,D_0,chi,delta) * (t_SNR[:,np.newaxis]*yrs - newescape(E,t_sed,p_max,t_max,p_min,t_b1,t_b2,p_b1,p_b2)) / pcm**2
    return result.T

def newR_ct(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
             p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,
            t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    c = C
    result = c * (t_SNR[:,np.newaxis]*yrs - newescape(E,t_sed,p_max,t_max,p_min,t_b1,t_b2,p_b1,p_b2)) /pcm
    return result.T

def newexcess(jumplim=0.05, maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,
            t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(t_sed,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    norm = 1
    accepted = 0
    peaks = 0
    peaks1 = 0
    peaks2 = 0
    peaks3 = 0
    peaks4 = 0
    peaks5 = 0
    for d in dist0:
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
        ratio = (g1.T / norm).T
        ratio1 = np.where(rd1>0, ratio, 1) - 1
        dif = np.gradient(ratio1, axis=0)
        maxr = np.argmax(dif, axis=0)
        mean = np.max(dif,axis=0)
        peak = np.logical_and(mean>jumplim, True)
        peak1 = np.logical_and.reduce([mean>jumplim, maxr<40])
        peak2 = np.logical_and.reduce([mean>jumplim, maxr<80, maxr>=40])
        peak3 = np.logical_and.reduce([mean>jumplim, maxr<120, maxr>=80])
        peak4 = np.logical_and.reduce([mean>jumplim, maxr<160, maxr>=120])
        peak5 = np.logical_and(mean>jumplim, maxr>=160)
        accepted += len(mean)
        peaks += np.sum(peak)
        peaks1 += np.sum(peak1)
        peaks2 += np.sum(peak2)
        peaks3 += np.sum(peak3)
        peaks4 += np.sum(peak4)
        peaks5 += np.sum(peak5)
    return np.array([accepted,peaks,peaks1,peaks2,peaks3,peaks4,peaks5])

def breakexcess(sigma=1, stat=0, maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,
            t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(0,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    norm = 1 / 17364.8109 * 26857.4584
    accepted = 0
    peaks = 0
    peaks1 = 0
    peaks2 = 0
    peaks3 = 0
    if stat == 0:
        s1 = 0.01 * np.ones(61)
        s2 = 0.1 * np.ones(139)
        jumplim = np.append(s1,s2)
    elif stat == 1:
        power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
        jumplim = 1e-4 * (E/1e-3) ** power
    for d in dist0:
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
        ratio = (g1.T / norm).T
        ratio1 = np.where(rd1>0, ratio, 1) - 1
        dif = np.diff(ratio1, axis=0)
        maxr = np.argmax(dif, axis=0)
        mean = np.max(dif,axis=0)
        peak1 = np.logical_and.reduce([mean>sigma*jumplim[maxr], maxr<=70])
        peak2 = np.logical_and.reduce([mean>sigma*jumplim[maxr], maxr>=71, maxr<=92])
        peak3 = np.logical_and.reduce([mean>sigma*jumplim[maxr], maxr>=93, maxr<154])
        accepted += len(mean)
        peaks = peaks + np.sum(peak1) + np.sum(peak2) + np.sum(peak3)
        peaks1 += np.sum(peak1)
        peaks2 += np.sum(peak2)
        peaks3 += np.sum(peak3)
    return np.array([accepted,peaks,peaks1,peaks2,peaks3])

def breakexcess2x( maxdist=300, maxage=3e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_SNR_II, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=1e5, p_min=1e-3,
            t_b1=1e4, t_b2=5e4, p_b1=1e-1, p_b2=1e1):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    ages0 = np.linspace(0,maxage,Nd)
    E = np.logspace(-3,np.log10(3e3),200)
    norm = 1  * D_0 / DIFFUSION_COEFFICIENT # / 12283.553742155571 * 11826.695987081592
    accepted = 0
    s1 = 0.01 * np.ones(61)
    s2 = 0.1 * np.ones(139)
    jumplimsys = np.append(s1,s2)
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = 1e-4 * (E/1e-3) ** power / 1.5
    sigmas = 0.1 + 0.2 * np.arange(26)
    resultsys = np.zeros(26)
    resultstat = np.zeros(26)
    for d in dist0:
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
        ratio = (g1.T / norm).T
        ratio1 = np.where(rd1>0, ratio, 0)
        dif = np.diff(ratio1, axis=0)
        maxr = np.argmax(dif, axis=0)
        mean = np.max(dif,axis=0)
        accepted += len(mean)
        for i,sigma in enumerate(sigmas):
            resultstat[i] += np.sum(np.logical_and.reduce([mean>sigma*jumplimstat[maxr], maxr<=154]))
            resultsys[i] += np.sum(np.logical_and.reduce([mean>sigma*jumplimsys[maxr], maxr<=154]))
    return np.append(resultsys / accepted, resultstat / accepted)

def excessbins(bins=200, maxdist=GALACTIC_RADIUS/10, maxage=2e5,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            t_b1=99999,t_b2=0, p_b1=10, p_b2=0):
    Nd=1500
    distsq = np.linspace(0,maxdist**2,5*Nd)
    ages0 = np.linspace(0,maxage,Nd)
    E = np.logspace(-3,2,bins)
    #norm = NORMAL4  * D_0 / DIFFUSION_COEFFICIENT # / 12283.553742155571 * 11826.695987081592
    normalisation1 = normalisation
    norm = 1 / (np.pi * D(E,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1
    jumps = 0
    accepted = 0
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = 1e-4 * (E/1e-3) ** power / 1.5
    for d2 in distsq:
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd)
        ratio = (g1.T / norm).T
        ratio1 = np.where(rd1>0, ratio, 0)
        dif = np.diff(ratio1, axis=0)
        maxr = np.argmax(dif, axis=0)
        mean = np.max(dif,axis=0)
        accepted += len(mean)
        jumps += np.sum(mean>jumplimstat[maxr])
    return jumps / accepted

def breakexcessradial(bins=200, maxdist=HALO_HEIGHT/2, maxage=2*SOURCE_LIFETIME,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    Nd=1500
    binsi = int(bins)
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    sources = np.array([sunintegral(r) for r in dist0])
    numbsources = np.diff(sources,prepend=0)
    sourcesinrange = sources[-1] * maxage
    ages0 = np.linspace(0,maxage,Nd)
    E = np.logspace(-3,2,binsi)
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation1 = normalisation
    norm = 1 / (np.pi * D(E,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1
    accepted = 0
    jumplimsys = np.where(E<10, 0.01, 0.1)[:-1]
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = (1e-4 * (E[1:]/1e-3) ** power / 1.5 + 1e-4 * (E[:-1]/1e-3) ** power / 1.5) /2
    sigmas = 0.1 + 0.2 * np.arange(26)
    resultsys = np.zeros(26)
    resultstat = np.zeros(26)
    for j,d2 in enumerate(distsq):
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd)
        ratio = (g1.T / norm)
        ratio1 = np.where(rd1.T>0, ratio, 0)
        dif = np.diff(ratio1, axis=1)
        mean = np.max(dif,axis=1)
        maxr = np.argmax(dif,axis=1)
        accepted += numbsources[j] * len(mean)
        for i,sigma in enumerate(sigmas):
            resultstat[i] += numbsources[j] * np.sum(np.any(dif>sigma*jumplimstat, axis=1))
            #print(int(10*sigma),np.sum(np.logical_and.reduce([mean>sigma*jumplimstat[maxr]])) - np.sum(np.any(dif>sigma*jumplimstat, axis=1)))
            resultsys[i] += numbsources[j] * np.sum(np.any(dif>sigma*jumplimsys, axis=1))
    return np.append(1 - (1 - resultsys / accepted) ** sourcesinrange, 1 - (1 - resultstat / accepted) ** sourcesinrange)

def breakexcessradial2(bins=200, maxdist=HALO_HEIGHT/2, maxage=SOURCE_LIFETIME,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            alpha = PROTON_INJECTION_SPECTRUM,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    Nd=1500
    bins = int(bins)
    binsi = 10*bins+1
    ages0 = np.linspace(0,maxage,Nd)
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    sources = np.array([sunintegral(r) for r in dist0])
    numbsources = np.diff(sources,prepend=0)
    sourcesinrange = sources[-1] * maxage
    lnE1 = np.linspace(np.log(1e-3),np.log(1e2),binsi)
    lnE2 = np.array([])
    lnE = lnE1
    E = np.exp(lnE)
    Edge = E[::10]
    gamma = alpha + delta
    ED = ((Edge[1:]**(1-gamma) - Edge[:-1]**(1-gamma)) / ((1-gamma) * (Edge[1:] - Edge[:-1]))) ** (-1/gamma)
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation1 = normalisation
    norm = 1 / (np.pi * D(ED,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * (Edge[1:] - Edge[:-1]) * ED**(-alpha)
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = 1e-4 * (Edge[1:-1]/1e-3) ** power / 1.5
    sigmas = 0.1 + 0.2 * np.arange(26)
    resultint = np.zeros(26)
    resultpoint = np.zeros(26)
    accepted = 0
    for j,d2 in enumerate(distsq):
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd) * E**(-alpha) *E
        G = np.where(rd1>0, g1, 0)
        gi1 = (G[:,1:] + G[:,:-1]) * np.diff(lnE) / 2
        gi = np.sum(gi1.reshape(Nd,bins,10),axis=2)
        rd2 = newR_dsq(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        rd3 = np.where(rd2>0, rd2, 1e-99)
        g2 = (np.pi*rd3)**(-3/2) * np.exp(-d2/rd3)* (Edge[1:] - Edge[:-1]) * ED**(-alpha)
        G2 = np.where(rd2>0, g2, 0)
        ratio1 = gi / norm
        ratio2 = G2 / norm
        dif1 = np.diff(ratio1,axis=1)
        dif2 = np.diff(ratio2,axis=1)
        accepted += numbsources[j] * len(np.any(dif1>jumplimstat, axis=1))
        for i,sigma in enumerate(sigmas):
            resultint[i] += numbsources[j] * np.sum(np.any(dif1>sigma*jumplimstat, axis=1))
            resultpoint[i] += numbsources[j] * np.sum(np.any(dif2>sigma*jumplimstat, axis=1))
    return np.append(1 - (1 - resultint / accepted) ** sourcesinrange, 1 - (1 - resultpoint / accepted) ** sourcesinrange)


def excessexperiment(maxdist=HALO_HEIGHT/2, maxage=SOURCE_LIFETIME,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            alpha = PROTON_INJECTION_SPECTRUM,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    Nd=1500
    ages0 = np.linspace(0,maxage,Nd)
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    sources = np.array([sunintegral(r) for r in dist0])
    numbsources = np.diff(sources,prepend=0)
    sourcesinrange = sources[-1] * maxage
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    lnE = np.log(E)
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation1 = normalisation
    normA = 1 / (np.pi * D(ED[:72],Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * (Edge[1:73] - Edge[:72]) * ED[:72]**(-alpha)
    normD = 1 / (np.pi * D(ED[72:],Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * (Edge[74:] - Edge[73:-1]) * ED[72:]**(-alpha)
    norm = np.concatenate((normA,normD))
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = np.concatenate((1e-4 * (Edge[1:72]/1e-3) ** power / 1.5 , 1e-4 * (Edge[74:-1]/1e-3) ** power / 1.5))
    sigmas = 0.1 + 0.2 * np.arange(26)
    resultint = np.zeros(26)
    resultpoint = np.zeros(26)
    accepted = 0
    for j,d2 in enumerate(distsq):
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        rd = np.where(rd1>0, rd1, 1e-99)
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd) * E**(-alpha) *E
        G = np.where(rd1>0, g1, 0)
        gi1 = (G[:,1:] + G[:,:-1]) * np.diff(lnE) / 2
        gi = np.concatenate((np.sum(gi1[:,:720].reshape(Nd,72,10),axis=2) ,
                             np.sum(gi1[:,721:].reshape(Nd,9,10),axis=2)), axis=1)
        rd2 = newR_dsq(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        rd3 = np.where(rd2>0, rd2, 1e-99)
        g2A = (np.pi*rd3[:,:72])**(-3/2) * np.exp(-d2/rd3[:,:72])* (Edge[1:73] - Edge[:72]) * ED[:72]**(-alpha)
        g2D = (np.pi*rd3[:,72:])**(-3/2) * np.exp(-d2/rd3[:,72:])* (Edge[74:] - Edge[73:-1]) * ED[72:]**(-alpha)
        G2A = np.where(rd3[:,:72]>0, g2A, 0)
        G2D = np.where(rd3[:,72:]>0, g2D, 0)
        G2 = np.concatenate((G2A,G2D),axis=1)
        ratio1 = gi / norm
        ratio2 = G2 / norm
        dif1 = np.diff(ratio1,axis=1)
        dif2 = np.diff(ratio2,axis=1)
        accepted += numbsources[j] * len(np.concatenate((np.any(dif1[:,:71]>jumplimstat[:71], axis=1),
                                                         np.any(dif1[:,72:]>jumplimstat[71:], axis=1))))
        for i,sigma in enumerate(sigmas):
            resultint[i] += numbsources[j] * np.sum(np.concatenate((np.any(dif1[:,:71]>sigma*jumplimstat[:71], axis=1),
                                                         np.any(dif1[:,72:]>sigma*jumplimstat[71:], axis=1))))
            resultpoint[i] += numbsources[j] * np.sum(np.concatenate((np.any(dif2[:,:71]>sigma*jumplimstat[:71], axis=1),
                                                         np.any(dif2[:,72:]>sigma*jumplimstat[71:], axis=1))))
    return np.append(1 - (1 - resultint / accepted) ** sourcesinrange, 1 - (1 - resultpoint / accepted) ** sourcesinrange)

def excessexperiment2(maxdist=HALO_HEIGHT/2, maxage=SOURCE_LIFETIME,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            alpha = PROTON_INJECTION_SPECTRUM,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    Nd=1500
    ages0 = np.linspace(0,maxage/2,Nd)
    distsq = np.linspace(0,maxdist**2,5*Nd)
    dist0 = np.sqrt(distsq)
    sources = np.array([sunintegral(r) for r in dist0])
    numbsources = np.diff(sources)
    distsq2 = (distsq[1:] + distsq[:-1]) / 2
    sourcesinrange = sources[-1] * maxage
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    c = LY_PC_CONVERSION
    normalisation1 = 0.000003457905225482213 * (np.pi * D(0.001,Dpcy, DIFFUSION_SPECTRUM))
    normall = 1 / (np.pi * D(E,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * E**(-alpha)
    norm = 1 / (np.pi * D(ED,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * ED**(-alpha)
    # normall = Amean(E) * E**(-alpha)
    # norm = Amean(ED) * ED**(-alpha)
    Elow = np.ones(len(E)-1)
    Ehigh = E[1:] - E[:-1] + 1
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = np.concatenate((1e-4 * (Edge[1:72]/1e-3) ** power / 1.5 , 1e-4 * (Edge[74:-1]/1e-3) ** power / 1.5))
    jumplimstat = np.where(np.delete(ED,(71,-1))<p_b1/2,1e10,jumplimstat)
    sigmas = 0.1 + 0.2 * np.arange(26)
    resultint = np.zeros(26)
    resultpoint = np.zeros(26)
    accepted = 0
    slopep = np.log(normall[1:]/normall[:-1]) / (np.log(Ehigh/Elow))
    normi1 = normall[:-1]/(slopep+1)*(Ehigh**(slopep+1)-Elow**(slopep+1))
    normi = np.concatenate((np.sum(normi1[:720].reshape(72,10),axis=1) ,
                             np.sum(normi1[721:].reshape(9,10),axis=1))) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    for j,d2 in enumerate(distsq2):
        rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        rd = np.where(rd1>0, rd1, 1e-99)
        ct1 = newR_ct(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        g1 = (np.pi*rd)**(-3/2) * np.exp(-d2/rd) * E**(-alpha)
        g11 = np.where(np.sqrt(d2)<ct1, g1, 0)
        G = np.where(rd1>0, g11, 0) + normall
        slope = np.log(G[:,1:]/G[:,:-1]) / (np.log(Ehigh/Elow))
        gi1 = np.where(slope==-1, G[:,:-1]*np.log(Ehigh/Elow) , G[:,:-1]/(slope+1)*(Ehigh**(slope+1)-Elow**(slope+1)))
        gi = np.concatenate((np.sum(gi1[:,:720].reshape(Nd,72,10),axis=2) ,
                             np.sum(gi1[:,721:].reshape(Nd,9,10),axis=2)), axis=1) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
        rd2 = newR_dsq(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        rd3 = np.where(rd2>0, rd2, 1e-99)
        ct2 = newR_ct(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
        g2 = (np.pi*rd3)**(-3/2) * np.exp(-d2/rd3) * ED**(-alpha)
        g21 = np.where(np.sqrt(d2)<ct2, g2, 0)
        G2 = np.where(rd2>0, g21, 0)
        ratio1 = (gi-normi) / normi
        ratio2 = (gi-normi) / norm
        dif1 = np.diff(ratio1,axis=1)
        dif2 = np.diff(ratio2,axis=1)
        accepted += numbsources[j] * len(np.any(np.delete(dif1,71,axis=1)>jumplimstat, axis=1))
        for i,sigma in enumerate(sigmas):
            resultint[i] += numbsources[j] * np.sum(np.any(np.delete(dif1,71,axis=1)>sigma*jumplimstat, axis=1))
            resultpoint[i] += numbsources[j] * np.sum(np.any(np.delete(dif2,71,axis=1)>sigma*jumplimstat, axis=1))
    return np.append(1 - (1 - resultint / accepted) ** sourcesinrange, 1 - (1 - resultpoint / accepted) ** sourcesinrange)

def debugger(d = 300, Nd = 1500,
             D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            alpha = PROTON_INJECTION_SPECTRUM,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    ages0 = np.linspace(0,t_max,int(Nd))
    approx_AMS = np.sqrt(1.16)*(1.10655)**((np.arange(721) - 5)/10)
    approx_DAMPE = 1983*(1.58308693)**((np.arange(91) - 5)/10)
    E = np.concatenate((approx_AMS, approx_DAMPE))/1000
    Edge = np.concatenate((E[:721:10],E[721::10]))
    ED = np.concatenate((E[5:722:10],E[726::10]))
    Elow = np.ones(len(E)-1)
    Ehigh = E[1:] - E[:-1] + 1
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation1 = 0.000003457905225482213 * (np.pi * D(0.001,Dpcy, DIFFUSION_SPECTRUM))
    normall = 1 / (np.pi * D(E,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * E**(-alpha)
    norm = 1 / (np.pi * D(ED,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * ED**(-alpha)
    slopep = np.log(normall[1:]/normall[:-1]) / (np.log(Ehigh/Elow))
    normi1 = normall[:-1]/(slopep+1)*(Ehigh**(slopep+1)-Elow**(slopep+1))
    normi = np.concatenate((np.sum(normi1[:720].reshape(72,10),axis=1) ,
                             np.sum(normi1[721:].reshape(9,10),axis=1))) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
    rd = np.where(rd1>0, rd1, 1e-99)
    ct1 = newR_ct(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
    g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd) * E**(-alpha)
    g11 = np.where(d<ct1, g1, 0)
    G = np.where(rd1>0, g11, 0) + normall
    slope = np.log(G[:,1:]/G[:,:-1]) / (np.log(Ehigh/Elow))
    gi1 = np.where(slope==-1, G[:,:-1]*np.log(Ehigh/Elow) , G[:,:-1]/(slope+1)*(Ehigh**(slope+1)-Elow**(slope+1)))
    gi = np.concatenate((np.sum(gi1[:,:720].reshape(Nd,72,10),axis=2) ,
                             np.sum(gi1[:,721:].reshape(Nd,9,10),axis=2)), axis=1) / (np.delete(Edge,73)[1:] - np.delete(Edge,73)[:-1])
    rd2 = newR_dsq(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
    rd3 = np.where(rd2>0, rd2, 1e-99)
    ct2 = newR_ct(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
    g2 = (np.pi*rd3)**(-3/2) * np.exp(-d**2/rd3) * ED**(-alpha)
    g21 = np.where(d<ct2, g2, 0)
    G2 = np.where(rd2>0, g21, 0)
    fluxi = np.sum(gi-normi,axis=0)
    fluxp = np.sum(G2,axis=0)
    return fluxi[-9:]

test = np.zeros((1000,9))
for i in range(1000):
    test[i] = debugger(4*i)

np.save('sumgreensintegral',test)
plt.plot(4*np.arange(1000),test)
plt.xlabel('distance [pc]')
plt.ylabel('relative error: (point-integrated)/sum')
plt.yscale('log')
plt.legend([int(a/1000) for a in approx_DAMPE])
plt.savefig('sumgreensintegral')

# print(sunintegral(30))
# test = excessexperiment2()
# print(test)
# print(testp)
# plt.plot(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,testi.T,label='integrated')
# plt.plot(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,testp.T,label='point')
# plt.xscale('log')
# plt.yscale('log')
# plt.title('greens functions / mean for different ages, integrated and unintegrated')
# plt.savefig('greenscomparison2')
# np.save('integratedmean2',test)
# distances0 = np.logspace(0,3,100)
# ages0 = np.logspace(3,6,100)
# ages1 = np.linspace(1e3,1e5,1000)
# lhaasobarsize = 25
# ranges = 27
# earr = np.arange(ranges)
# E0 = 0.01 * (1 + lhaasobarsize/100) ** (2*earr)
num_cores = 9

'''
test = Parallel(n_jobs=num_cores)(delayed(excessconstantage)(t,Emid,yerr=3*0.26,barsize=lhaasobarsize)
                                     for t,Emid in test)

result = np.array(test).reshape(ranges,1000)

plt.figure()
plt.title('probability of seeing a peak that is higher than error of measured flux')
plt.pcolor(E0,ages1,result.T)
plt.xlabel('Emid')
plt.ylabel('age')
plt.yscale('log')
plt.xscale('log')
plt.colorbar()
#plt.savefig('prob_2d_constdist_lin.png')
#
plt.show()
E0only = np.zeros(ranges)
for i in range(ranges):
    E0only[i] = np.mean(result[i,:])

plt.figure()
plt.plot(E0,E0only)
plt.xscale('log')
plt.xlabel('Energy')
plt.ylabel('probability')
plt.savefig('prob_E0_3sigma.png')
print('probability of seeing excess at each energy bin is', E0only)
print('probability of seeing excess in total', np.sum(E0only)) ### minimal 0.469289  ### 3 sigma 0.10497899999999999

'''
N = 45
breakE = np.logspace(-2.9,3.5,N)
breakt = np.logspace(3.2,4.9,N)
breakt = np.append(np.array([0]),breakt)
ages = np.logspace(4.5,6.3,N)
ages1 = np.logspace(3.2,6,N)
sed = np.logspace(-1,5,N)
slope = releseslope(ages)
slopeog = releseslope(1e5)
acc = np.zeros(N)
pe = np.zeros(N)
excesses = np.linspace(0,0.01,N)
sigmas = np.linspace(0.5,5,N)
distances = np.linspace(100,10000,N)
'''runy = 36
flux = np.logspace(-10,0,runy)
accepted = np.zeros(runy)
peaks = np.zeros(runy)
nopeak = np.zeros(ranges)

printProgressBar(0, N, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i in range(N):
    result = np.array(Parallel(n_jobs=num_cores)(delayed(excessall)(E,barsize=lhaasobarsize,excesslim=7,life=ages[i],beta=slope[i])
                                    for E in E0))d
    accepted = result[:,0]
    peaks = result[:,1]
    printProgressBar(i + 1, N, prefix = 'Progress:', suffix = 'Complete', length = 50)
    acc[i] = np.sum(accepted)
    pe[i] = np.sum(peaks)
print('greens',greensnppeak(Ea,4e3,50))'''

maxage = 2*SOURCE_LIFETIME
maxdist = HALO_HEIGHT/2
power = 3*maxage/100*(maxdist**2/GALACTIC_RADIUS**2)

#numbers = np.array(Parallel(n_jobs=num_cores)(delayed(breakexcess)(sigma=g,stat=1,maxdist=maxdist,maxage=maxage,D_0=DIFFUSION_COEFFICIENT,t_max=1e5,t_b1=99999,t_b2=0,p_b1=10)for g in breakE)).T
#result = np.array(Parallel(n_jobs=num_cores)(delayed(compare2d)(maxdist=maxdist,maxage=maxage,beta = g)for g in slope))
#numbers = newexcess()

binnumber = 9 * (np.arange(45) + 1)
#result = np.array(Parallel(n_jobs=num_cores)(delayed(excessbins)(bins=int(g))for g in slope))
# result = np.zeros(len(binnumber))
# for i in range(len(result)):
#     result[i] = excessbins(bins=binnumber[i])
#     print(binnumber[i])
# print(result)
# plt.plot(binnumber,result)
# plt.xlabel('number of bins from 1GV to 100TV')
# plt.ylabel('p_1')
# plt.savefig('bintesting.png')
# prob0 = np.zeros((25,N))
# prob1 = np.zeros((25,N))
# prob2 = np.zeros((25,N))
# prob3 = np.zeros((25,N))
# prob4 = np.zeros((25,N))
# prob5 = np.zeros((25,N))
# prob6 = np.zeros((25,N))
# prob7 = np.zeros((25,N))
# for i in range(25):
#     print(i)
#     jl = i/5 + 0.5
#     numbers = np.array(Parallel(n_jobs=num_cores)(delayed(breakexcess2x)(sigma=jl,maxdist=maxdist,maxage=maxage,D_0=DIFFUSION_COEFFICIENT,t_max=1e5,t_b1=99999,t_b2=0,p_b1=g)for g in breakE)).T
#     prob0[i] = 1 - (1 - numbers[1] / numbers[0]) ** power
#     prob1[i] = 1 - (1 - numbers[2] / numbers[0]) ** power
#     prob2[i] = 1 - (1 - numbers[3] / numbers[0]) ** power
#     prob3[i] = 1 - (1 - numbers[4] / numbers[0]) ** power
#     prob4[i] = 1 - (1 - numbers[5] / numbers[0]) ** power
#     prob5[i] = 1 - (1 - numbers[6] / numbers[0]) ** power
#     prob6[i] = 1 - (1 - numbers[7] / numbers[0]) ** power
#     prob7[i] = 1 - (1 - numbers[8] / numbers[0]) ** power
# np.savetxt('break_jump_sys_highH_allnew',prob0)
# np.savetxt('break_jump_sys_highH_lownew',prob1)
# np.savetxt('break_jump_sys_highH_midnew',prob2)
# np.savetxt('break_jump_sys_highH_highnew',prob3)
# np.savetxt('break_jump_stat_highH_allnew',prob4)
# np.savetxt('break_jump_stat_highH_lownew',prob5)
# np.savetxt('break_jump_stat_highH_midnew',prob6)
# np.savetxt('break_jump_stat_highH_highnew',prob7)
# results = np.zeros((20,52))
# for i in range(20):
#     results[i] = breakexcessradial(maxdist=(1+i)*200)
#     print(i)

#np.savetxt('max_dist',results)
maxdists = np.array([100,200,300,400,500,600,1000,2000,4000])
#prob = np.array(Parallel(n_jobs=num_cores)(delayed(excessexperiment2)(maxdist=g)for g in maxdists))
#prob = breakexcessradial(bins=20)
#breakexcessradial()
#np.savetxt('distmid',prob)
# pb = 10**(-2)
# def greensjump(E, t_SNR, d, t_max=1e5,t_b1=99999,t_b2=0,p_b1=pb):
#     rd1 = newR_dsq(E, t_SNR, t_max=t_max, t_b1=t_b1, t_b2=t_b2, p_b1=p_b1)
#     rd = np.where(rd1>0, rd1, 1e-99)
#     g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd)
#     return g1
'''
default_cycler = (cycler(color=['r', 'g', 'b','c']))
#fig = plt.figure(figsize=(14.20, 10.80))

plt.rc('font', size=22)

plt.rc('axes', prop_cycle=default_cycler)
da = np.array([30,300,1000])
ta = np.array([2000,10000,50000,110000])

fig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(23,15))
#da = np.array([700,800,900])
#ta = np.array([60000,70000,80000,90000])
tas = (np.meshgrid(ta,da)[0].flatten()/1000).astype('int')  #+ np.array(['1'])
das = (np.meshgrid(ta,da)[1].flatten()).astype('str')
#legend = np.array([das[i] + 'pc, ' + str(tas[i]) + 'kyr' for i in range(len(das))])

axs[0,0].loglog(Ea,greensjump(Ea,ta,da[0],p_b1=1)*D(Ea[:,np.newaxis])**(3/2),linestyle='--',linewidth=3)
axs[0,0].loglog(Ea,greensjump(Ea,ta,da[1],p_b1=1)*D(Ea[:,np.newaxis])**(3/2),linestyle='-.',linewidth=3)
axs[0,0].loglog(Ea,greensjump(Ea,ta,da[2],p_b1=1)*D(Ea[:,np.newaxis])**(3/2),linestyle='dotted',linewidth=3)



axs[0,1].loglog(Ea,greensjump(Ea,ta,da[0],p_b1=10)*D(Ea[:,np.newaxis])**(3/2),linestyle='--',linewidth=3)
axs[0,1].loglog(Ea,greensjump(Ea,ta,da[1],p_b1=10)*D(Ea[:,np.newaxis])**(3/2),linestyle='-.',linewidth=3)
axs[0,1].loglog(Ea,greensjump(Ea,ta,da[2],p_b1=10)*D(Ea[:,np.newaxis])**(3/2),linestyle='dotted',linewidth=3)



axs[1,0].loglog(Ea,greensjump(Ea,ta,da[0],p_b1=100)*D(Ea[:,np.newaxis])**(3/2),linestyle='--',linewidth=3)
axs[1,0].loglog(Ea,greensjump(Ea,ta,da[1],p_b1=100)*D(Ea[:,np.newaxis])**(3/2),linestyle='-.',linewidth=3)
axs[1,0].loglog(Ea,greensjump(Ea,ta,da[2],p_b1=100)*D(Ea[:,np.newaxis])**(3/2),linestyle='dotted',linewidth=3)



axs[1,1].loglog(Ea,greensjump(Ea,ta,da[0],p_b1=1000)*D(Ea[:,np.newaxis])**(3/2),linestyle='--',linewidth=3)
axs[1,1].loglog(Ea,greensjump(Ea,ta,da[1],p_b1=1000)*D(Ea[:,np.newaxis])**(3/2),linestyle='-.',linewidth=3)
axs[1,1].loglog(Ea,greensjump(Ea,ta,da[2],p_b1=1000)*D(Ea[:,np.newaxis])**(3/2),linestyle='dotted',linewidth=3)
for ax in axs.flat:
    ax.set_ylim(1e34,1e41)
axs[0,0].set_ylabel('$R^{0.4}$ G [arb. u.]')
axs[1,0].set_ylabel('$R^{0.4}$ G [arb. u.]')
axs[1,0].set_xlabel('R [TV]')
axs[1,1].set_xlabel('R [TV]')

fig.tight_layout()
plt.savefig('greens2x2high.png',doi=1500,bbox_inches='tight')
'''
# fig = plt.figure(figsize=(10,10))
# ta = np.logspace(4,5,20)
# plt.loglog(Ea,NORMAL4*D(Ea)**(3/2),c='black',linewidth=3,linestyle='--')
# plt.loglog(Ea,greensjump(Ea,ta,300,p_b1=100)*D(Ea[:,np.newaxis])**(3/2),linewidth=2)
# plt.ylim(1e34,1e38)
# fig.tight_layout()
# plt.savefig('greensproblem.png',doi=1500,bbox_inches='tight')
'''
plt.plot(Ea,greensjump(Ea,ta,da[0])*D(Ea[:,np.newaxis])**(3/2),linestyle='--',linewidth=3)
plt.plot(Ea,greensjump(Ea,ta,da[1])*D(Ea[:,np.newaxis])**(3/2),linestyle='-.',linewidth=3)
plt.plot(Ea,greensjump(Ea,ta,da[2])*D(Ea[:,np.newaxis])**(3/2),linestyle='dotted',linewidth=3)
# plt.plot(0,0,linestyle='--',color='black',label='30pc')
# plt.plot(0,0,linestyle='-.',color='black',label='300pc')
# plt.plot(0,0,linestyle='dotted',color='black',label='1000pc')
# plt.plot(0,0,color='r',label='2kyr')
# plt.plot(0,0,color='g',label='10kyr')
# plt.plot(0,0,color='b',label='50kyr')
# plt.plot(0,0,color='c',label='110kyr')
plt.plot(Ea,greensnppeak(Ea[:,np.newaxis],ta,da[0])*D(Ea[:,np.newaxis])**(3/2),linestyle='--',linewidth=3,alpha=0.3)
plt.plot(Ea,greensnppeak(Ea[:,np.newaxis],ta,da[1])*D(Ea[:,np.newaxis])**(3/2),linestyle='-.',linewidth=3,alpha=0.3)
plt.plot(Ea,greensnppeak(Ea[:,np.newaxis],ta,da[2])*D(Ea[:,np.newaxis])**(3/2),linestyle='dotted',linewidth=3,alpha=0.3)
plt.xlabel('R [TV]')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$R^{0.4}$ G [arb. u.]')
plt.ylim(1e34,1e41)
#plt.legend(loc="center right", bbox_to_anchor=(1.22, 0.5))
#plt.legend(legend,loc="center right", bbox_to_anchor=(1.35, 0.5), ncol=1)
#plt.title('Break Rigidity ' + str(pb) + 'TV')
plt.savefig('greensoverlay.png',doi=1500,bbox_inches='tight')
#plt.savefig('greensjump' + str(pb) + '.png',doi=1500,bbox_inches='tight')
'''
# pbs = np.logspace(-2.9999,3,7)
# colors = plt.cm.plasma(np.linspace(0,0.5,7))
# for i, pb in enumerate(pbs):
#     plt.plot(Ea,newescape(Ea[:,np.newaxis],t_max=1e5,t_b1=99999,t_b2=0,p_b1=pb)/YEAR_SECOND_CONVERSION/1000,color=colors[i])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Rigidity [TV]')
# plt.ylabel('Escape Time [kyr]')
# plt.ylim(1.6,150)

#plt.savefig('escapetime.png',doi=1500,bbox_inches='tight')

# h224Dt = np.logspace(-15,10,1000000)

# def jacobitheta(a):
#     loops = 1000
#     result = 1
#     q = np.exp(-a)
#     G = 1
#     for i in range(loops):
#         G *= (1-q**(2*(i+1)))
#     for j in range(loops):
#         result *= (1-2*q**(2*i+1)+q**(4*(i+1)-2))
#     return result * G

# j = jacobitheta(h224Dt)
# print(np.max(j),min(j))
# plt.loglog(h224Dt,j)
# plt.savefig('test2')
# thetafunction = np.vstack((h224Dt,j))
# np.save('thetafunction',thetafunction)


# normall = 1 / (np.pi * D(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,Dpcy, DIFFUSION_SPECTRUM)) * normalisation * AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY**(-PROTON_INJECTION_SPECTRUM)
# np.save('mean',normall)

def int2(R, theta,E):
    pointsofinterest = [0,1,10,1e2,1e3,1e4,1e5,1e6]
    return quad(integrant,0,1e15,args=(R,theta,E),points=pointsofinterest,epsrel=1e-20)[0]

def int3(theta,E):
    pointsofinterest = SUN_POSITION + 600*(np.arange(11)-5)
    return quad(int2,0,GALACTIC_RADIUS,args=(theta,E),points=pointsofinterest,epsrel=1e-20)[0]




# mean = np.array([quad(int3,0,2*np.pi,args=(n,),epsrel=1e-20) for n in AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY])
# print(mean)
# np.save('mean4',mean)
# normall = 1 / (np.pi * D(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,Dpcy, DIFFUSION_SPECTRUM)) * normalisation
# mean2 = np.array([mean_bessel(E) for E in AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY])
# print(mean2)
# np.save('meani',mean2)
# plt.loglog(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,mean[:,0],label='triple')
# plt.loglog(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,mean2[:,0],label='bessel')
# plt.plot(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,0.000003457905225482213*D(0.001)/D(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)/AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY*np.sqrt((AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)**2+M_PROTON**2)/(meanA),label='Khai/Anton, with beta')
# plt.plot(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY,np.sqrt(2)*0.000003457905225482213*D(0.001)/D(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)/(meanA),label='Khai/Anton, without beta')
# plt.title('comparing the means')
# plt.xscale('log')
# plt.legend()
# a1 = (np.sqrt(2)*0.000003457905225482213*D(0.001)/D(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)/(meanA))[0]
# a2 = (np.sqrt(2)*0.000003457905225482213*D(0.001)/D(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)/AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY*np.sqrt((AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY)**2+M_PROTON**2)/(meanA))[0]
# print(a1,a2,a2/a1,a1/a2)
# print(AMS_DAMPE_DEFAULT_RIGIDITY_ARRAY[0])
# plt.savefig('mean_comparison4')
# t0 = np.logspace(2,12,10000)
# R0 = np.linspace(0,15000,5)
# for R in R0:
#     plt.loglog(t0,integrant2(t0,R,1e3))
# plt.savefig('test35')


print('runtime' , timeit.default_timer() - start)
print('runtime' , datetime.now() - startt)

# -3 -2 -1 0 1 2 3
# 6 5 4 4 3 3 2
# 12 12 11 10 10 9 9