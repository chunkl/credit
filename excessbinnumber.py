import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import tplquad, quad, quad_vec, dblquad
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.colors as colors

#time the code
startt = datetime.now()
c = 3e10 #cm/s
YEAR_SECOND_CONVERSION = 31536000
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
DIFFUSION_SPECTRUM = 0.6
DIFFUSION_COEFFICIENT = 6.7825E27 # cm2 s-1
ISM_PARTICLE_DENSITY = 1
MAXIMUM_PARTICLE_MOMENTUM = 3E3 #TeV
PROTON_INJECTION_SPECTRUM = 2.2

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

def greensfactor(R,theta,r_0=SUN_POSITION,H=HALO_HEIGHT,n=0):
    return R / np.sqrt((R*np.cos(theta)-r_0)**2 + R**2*np.sin(theta)**2 + (2*n*H)**2)

def integrant(R, theta, r_0=SUN_POSITION, H=HALO_HEIGHT, n=0):
    return greensfactor(R,theta,r_0,H,n)*FerrierSource(R,r_0)

def newescape(E, t_sed=SEDOV_TIME,
             p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=1e-3,
            t_b1=SOURCE_LIFETIME-1, t_b2=0, p_b1=10, p_b2=0):
    yrtos = YEAR_SECOND_CONVERSION
    if t_b2==0:
        s1 = np.log(t_b1/t_sed) / np.log(p_max/p_b1)
        s2 = np.log(t_max/t_b1) / np.log(p_b1/p_min)
        t1 = yrtos * t_b1 * (p_b1 / E[E<p_b1]) ** s2
        t2 = yrtos * t_sed * (p_max / E[E>=p_b1]) ** s1
        result = np.where(E<p_b1, yrtos * t_b1 * (p_b1 /E) ** s2, yrtos * t_sed * (p_max / E) ** s1)
        return result
    else:
        s1 = np.log(t_b1/t_sed) / np.log(p_max/p_b2)
        s2 = np.log(t_b2/t_b1) / np.log(p_b2/p_b1)
        s3 = np.log(t_max/t_b2) / np.log(p_b1/p_min)
        t1 = yrtos * t_b2 * (p_b1 / E[E<p_b1]) ** s3
        t2 = yrtos * t_b1 * (p_b2 / E[(E<p_b2) & (E>p_b1)]) ** s2
        t3 = yrtos * t_sed * (p_max / E[E>p_b2]) ** s1
        return np.append(np.append(t1,t2),t3)

def newR_dsq(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
             p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=1e-3,
            t_b1=SOURCE_LIFETIME-1, t_b2=0, p_b1=10, p_b2=0):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    result = 4 * D(E,n,D_0,chi,delta) * (t_SNR[:,np.newaxis]*yrs - newescape(E,t_sed,p_max,t_max,p_min,t_b1,t_b2,p_b1,p_b2)) / pcm**2
    return result.T

def newR_dsqstar(E, t_SNR, D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
             p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=1e-3,
            t_b1=SOURCE_LIFETIME-1, t_b2=0, p_b1=10, p_b2=0):
    yrs = YEAR_SECOND_CONVERSION
    pcm = CM_PC_CONVERSION
    tesc = newescape(E,t_sed,p_max,t_max,p_min,t_b1,t_b2,p_b1,p_b2).reshape(np.shape(E))
    result = 4 * D(E,n,D_0,chi,delta) * (t_SNR[:,np.newaxis]*yrs - tesc) / pcm**2
    return result.T

#galacticintegral = np.array([dblquad(integrant,0,2*np.pi,0,GALACTIC_RADIUS, args=(SUN_POSITION, HALO_HEIGHT, n))[0] for n in range(1000001)])
#np.savetxt('galacticintegral',galacticintegral)
galacticintegral = np.loadtxt('galacticintegral')

conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
Dpcy = DIFFUSION_COEFFICIENT * conversion
normalisation = galacticintegral[0] + 2*np.sum((-1)**np.arange(1000001)[1:]*galacticintegral[1:])

def breakexcessradial(bins=200, dist=300, maxdist=HALO_HEIGHT/2, maxage=2*SOURCE_LIFETIME,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    Nd=15
    binsi = int(bins)
    ages0 = np.linspace(0,maxage,Nd)
    E = np.logspace(-3,2,binsi)
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation1 = normalisation
    norm = 1 / (np.pi * D(E,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = (1e-4 * (E[1:]/1e-3) ** power / 1.5 + 1e-4 * (E[:-1]/1e-3) ** power / 1.5) /2
    rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
    rd = np.where(rd1>0, rd1, 1e-99)
    g1 = (np.pi*rd)**(-3/2) * np.exp(-dist**2/rd)
    ratio = (g1.T / norm)
    ratio1 = np.where(rd1.T>0, ratio, 0)
    dif = np.diff(ratio1, axis=1)
    return g1

def breakexcessradial2(bins=200, dist=100, maxdist=HALO_HEIGHT/2, maxage=2*SOURCE_LIFETIME,
            D_0=DIFFUSION_COEFFICIENT, delta=DIFFUSION_SPECTRUM, t_sed=SEDOV_TIME, n=1, chi=1,
            p_max=MAXIMUM_PARTICLE_MOMENTUM, t_max=SOURCE_LIFETIME, p_min=MIN_RIGIDITY,
            alpha = PROTON_INJECTION_SPECTRUM,
            t_b1=SOURCE_LIFETIME-1,t_b2=0, p_b1=10, p_b2=0):
    Nd=1500
    bins = int(bins)
    binsi = 10*bins+1
    ages0 = np.linspace(0,maxage,Nd)
    lnE1 = np.linspace(np.log(1e-3),np.log(1e2),binsi)
    lnE2 = np.array([])
    lnE = lnE1
    E = np.exp(lnE)
    Edge = E[::10]
    gamma = alpha + delta
    ED = ((Edge[1:]**(1+gamma) - Edge[:-1]**(1+gamma)) / ((1+gamma) * (Edge[1:] - Edge[:-1]))) ** (1/gamma)
    conversion = YEAR_SECOND_CONVERSION/CM_PC_CONVERSION**2
    Dpcy = D_0 * conversion
    normalisation1 = normalisation
    norm = 1 / (np.pi * D(ED,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * (Edge[1:] - Edge[:-1]) * ED**(-alpha)
    power = np.log(1e-1/1e-4) / np.log(1e2/1e-3)
    jumplimstat = 1e-4 * (Edge[1:-1]/1e-3) ** power / 1.5 
    rd1 = newR_dsq(E, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
    rd = np.where(rd1>0, rd1, 1e-99)
    g1 = (np.pi*rd)**(-3/2) * np.exp(-dist**2/rd) * E**(-alpha) *E
    G = np.where(rd1>0, g1, 0) 
    # g1 = (-3/2 * np.log(np.pi*rd) - dist**2/rd).T - alpha * np.log(E)
    # gl = np.where(rd1.T>0, g1 , -1e99)
    # slope = (gl[:,1:] - gl[:,:-1]) / (np.log(E)[1:] - np.log(E)[:-1])
    # n = gl[:,:-1] - slope * np.log(E)[:-1]
    # gi = np.where(slope==-1, np.exp(n)*np.log(E[1:]/E[:-1]) , np.exp(n)/(slope+1)*(E[1:]**(slope+1)-E[:-1]**(slope+1)))
    # gi2 = gi.reshape(Nd,10,bins)
    # gi3 = np.sum(gi2,axis=1)
    # Egstar = np.where(slope==-1 , ((E[1:]**(1+slope)-E[:-1]**(1+slope)) / ((1+slope)*(E[1:]-E[:-1]))**(1/slope)) , (E[1:]-E[:-1])/np.log(E[1:]/E[:-1]))
    # rd1star = newR_dsqstar(Egstar, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2)
    # rdstar = np.where(rd1star>0, rd1star, 1e-99)
    # g1star = (np.pi*rdstar)**(-3/2) * np.exp(-dist**2/rdstar)
    # gstar = np.where(rd1star>0, g1star, 0).T * Egstar ** (-alpha) * (E[1:]-E[:-1])
    # gstar1 = gstar.reshape(Nd,10,bins)
    # gi = np.sum(gstar1,axis=1)
    # ratio = gi / norm
    gi1 = (G[:,1:] + G[:,:-1]) * np.diff(lnE) / 2
    gi = np.sum(gi1.reshape(Nd,bins,10),axis=2)
    rd2 = newR_dsq(ED, ages0, D_0, delta, t_sed, n, chi, p_max, t_max, p_min, t_b1, t_b2, p_b1, p_b2).T
    rd3 = np.where(rd2>0, rd2, 1e-99)
    g2 = (np.pi*rd3)**(-3/2) * np.exp(-dist**2/rd3)* (Edge[1:] - Edge[:-1]) * ED**(-alpha)
    G2 = np.where(rd2>0, g2, 0) 
    ratio1 = gi / norm
    ratio2 = G2 / norm
    dif1 = np.diff(ratio1,axis=1)
    dif2 = np.diff(ratio2,axis=1)
    jumps1 = np.sum(np.any(dif1>jumplimstat, axis=1))
    jumps2 = np.sum(np.any(dif2>jumplimstat, axis=1))
    # ratio = (g1.T / norm)
    # ratio1 = np.where(rd1.T>0, ratio, 0)
    # dif = np.diff(ratio1, axis=1)
    return jumps1, jumps2
Nb = 114
d = 550

# plt.plot(np.logspace(-3,2,Nb),breakexcessradial2(Nb,dist=d).T)
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('$(G_{integrated} - G)$')
# plt.ylim(1e-5,1e1)
# plt.xlabel('R [TV]')
# plt.hlines(1,xmin=1e-3,xmax=1e2,linestyle='--',color='black')
# plt.title(str(Nb) + 'bins between 1GV and 100TV, integrated' + str(d) + 'pc difference to point')
# plt.savefig('greens'+str(Nb)+'integrated' + str(d) + 'pcdiff')


# N = 150
# Ea = np.logspace(-3,2,N)
# d = 1000
# ages0 = np.logspace(3,np.log10(2e5),30)
# Dpcy = DIFFUSION_COEFFICIENT * conversion
# normalisation1 = normalisation
# norm = 1 / (np.pi * D(Ea,Dpcy, DIFFUSION_SPECTRUM)) * normalisation1 * Ea**0.6
# rd1 = newR_dsq(Ea, ages0)
# rd = np.where(rd1>0, rd1, 1e-99).T
# g1 = (np.pi*rd)**(-3/2) * np.exp(-d**2/rd) * Ea**0.6

# plt.loglog(Ea,norm,color='black',linestyle='--')
# plt.loglog(Ea,g1.T)
# plt.ylim(1e-14,1e-5)
# plt.title(str(d) + 'pc' + str(N) + 'bins' )
# plt.savefig('greens' + str(d) + 'pc' + str(N) + 'bins')



# binnumbers = np.linspace(10,114,100)
# num_cores = 9
# distances = np.array([500,550,600,650,700,725,750,775,800])
# colorss = plt.cm.jet(np.linspace(0, 1, len(distances)))
# for i,d in enumerate(distances):
#     numbers = np.array(Parallel(n_jobs=num_cores)(delayed(breakexcessradial2)(bins=g,dist=d,maxage=4*SOURCE_LIFETIME) for g in binnumbers))
#     plt.plot(binnumbers,numbers[:,0],label=d,linestyle='--',c=colorss[i])
#     plt.plot(binnumbers,numbers[:,1],linestyle='dotted',c=colorss[i])
# plt.legend(ncol=3,loc='upper left')
# plt.xlabel('number of bins')
# plt.ylabel('number of jumps for 1500 sources')
# plt.ylim(1000,1300)
# plt.savefig('bintestintegrated5')
n=65.31386851031755
sigmas = 0.1 + 0.2 * np.arange(26)
integration1 = np.array([1.         ,0.99999183, 0.99965766, 0.99787852, 0.99357103, 0.98631194,
 0.97622593, 0.96364759, 0.94909936, 0.93303758, 0.91586037, 0.89776043,
 0.87919309, 0.8603693,  0.84153447, 0.82274501, 0.80421902, 0.78594168,
 0.76798453, 0.750288,   0.73279113, 0.71555294, 0.69882135, 0.68257051,
 0.66662462, 0.65123574, ])
point1 = np.array([1.       ,  0.9999967,  0.99986011, 0.99911312,
 0.9972125,  0.99380184, 0.98870486, 0.98187296, 0.97330426, 0.96303066,
 0.95109592, 0.93752847, 0.92234239, 0.90555797, 0.88759965, 0.86872816,
 0.84931602, 0.82958668, 0.80971799, 0.78997771, 0.77042766, 0.75120022,
 0.73241854, 0.71408145, 0.69624937, 0.67888436])
integration2 = np.array([0.99999992, 0.99993719, 0.99911993, 0.99627768, 0.99059003, 0.9819205,
 0.97052411, 0.95683933, 0.94139347, 0.92460773, 0.90674692, 0.88822634,
 0.86929915, 0.8502773,  0.83127897, 0.81242555, 0.79377993, 0.77544506,
 0.75735015, 0.73943815, 0.72169353, 0.70449723, 0.687642,   0.67123706,
 0.65535123, 0.63994095, ])
point2 = np.array([1.     ,    0.9999967,  0.99986011, 0.99911312,
 0.9972125,  0.99380184, 0.98870486, 0.98187296, 0.97330426, 0.96303066,
 0.95109592, 0.93752847, 0.92234239, 0.90555797, 0.88759965, 0.86872816,
 0.84931602, 0.82958668, 0.80971799, 0.78997771, 0.77042766, 0.75120022,
 0.73241854, 0.71408145, 0.69624937, 0.67888436,])
corrected = np.load('smallermean.npy')
divide4 = np.load('integratedmean.npy')
divide3 = np.load('mybessel2.npy')
divide2 = np.load('integratedmean2.npy')
dividepi = np.load('divideby4timespi.npy')
Anton = np.load('Anton_integral_stat_10TV_100ykr_fiducial.npy')/100
sigmaA = np.linspace(0,5,101)
#print(divide4-divide2)
#plt.plot(sigmas,integration1,label='logspace integration')
#plt.plot(sigmas,point1-point2,label='point sampling 1')
plt.plot(sigmas,-np.gradient(divide4[:26],sigmas),label='integrated mean')
plt.plot(sigmas,-np.gradient(divide4[26:],sigmas),label='point mean')
plt.plot(sigmas,-np.gradient(divide2[:26],sigmas),label='changed Antons mean')
plt.plot(sigmas,-np.gradient(divide2[26:],sigmas),label='changed Antons mean points') 
plt.plot(sigmaA,-np.gradient(Anton,sigmaA),label='Antons probability')
# plt.plot(sigmas,divide4[:26],label='integrated mean')
# plt.plot(sigmas,divide4[26:],label='point mean')
# plt.plot(sigmas,divide2[:26],label='Antons mean')
# plt.plot(sigmas,divide2[26:],label='Antons mean points') 
# plt.plot(sigmaA,Anton,label='Antons probability')
#plt.ylim(0.9,1)
#plt.xlim(0,2)

plt.legend()
plt.title('negative of slope of p')
plt.savefig('sigmatest7')

# mean = np.load('mean.npy') * 1000**0.6
# meanA = np.load('mean_Anton.npy')
# print(np.mean(meanA/mean))
# plt.plot(meanA/mean)
# plt.savefig('meantest')

print('runtime:',datetime.now()-startt)


