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

startt = datetime.now()
ns = np.arange(1000000)
def numbersoure(R,r,d):
    if (d**2+R**2-r**2)/(2*R*d)>1:
        print('large cos')
    elif (d**2+R**2-r**2)/(2*R*d)<0:
        print('small cos')
    return 2*R*np.arccos((d**2+R**2-r**2)/(2*R*d))

def greensfactor(R,theta,r_0=8300,H=4000,n=0):
    return R / np.sqrt((R*np.cos(theta)-r_0)**2 + R**2*np.sin(theta)**2 + (2*n*H)**2)

def sunintegrant(R,r =10, r_0=8300):
    return 2 * R * np.arccos((r_0**2 + R**2 -r**2) / (2*R*r_0))

def radial(R, r_0=8300):
    result = 4.745 * np.exp(-(R-r_0)/4500)
    if R>3700:
        result += 26.679 * np.exp(-(R**2 - r_0**2)/(6800**2))
    else:
        result += 26.679 * np.exp(-(3700**2 - r_0**2)/(6800**2)) * np.exp(-((R - 3700)/2100)**2)
    return result

def radial1(R, r_0=8300):
    return 0

def radial2(R, r_0=8300, A=44, B=0.2, C=1.4):
    # 44,0.2,1.4
    # 41,1.9,5
    return A * (R/r_0)**B * np.exp(-C*(R-r_0)/r_0)

def radial3(R, a=1.068, R_exp=4500):
    return a * (R/R_exp**2) * np.exp(-R/R_exp)

def radial4(R, r_0=8300, A=1050, b=6.8, a=4):
    return A * (R/r_0)**a * np.exp(-b*R/r_0)

def radial5(R, r_0=8300, beta=3.53):
    A = beta**4 * np.exp(-beta) / (12*np.pi)
    return A / r_0**2 * (R/r_0)**2 * np.exp(-beta*(R-r_0)/r_0)

def integrant(R,theta,r_0=8300,H=4000,n=0):
    return radial2(R,r_0) * greensfactor(R,theta,r_0,H,n)

def sunintegrant2(R,r=10,r_0=8300,H=4000,n=0):
    return radial(R,r_0)*sunintegrant(R,r,r_0)

def sunintegral(r, r_0=8300):
    return quad(sunintegrant2,r_0-r,r_0+r,args=(r,r_0))


#print(dblquad(greensfactor,0,2*np.pi,0,15000)[0]/15000**2)
def rad(R,r_0=8300):
    return 2*np.pi*R*radial(R,r_0)

print(quad(rad,0,15000)[0]/1e4/1e6)
radial = np.vectorize(radial)
Rs = np.linspace(0,15000,100)
plt.plot(Rs,radial(Rs)/quad(rad,0,15000)[0])
plt.savefig('test.png')

#print(dblquad(integrant,0,2*np.pi,0,15000))
#print([sunintegral(r) for r in range(100)])
#a,_ = dblquad(integrant,0,2*np.pi,0,15000,args=(8300,4000,1,))

print('runtime' , datetime.now() - startt)