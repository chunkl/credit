import numpy as np
#from matplotlib import cm
from scipy.integrate import quad
from scipy.special import erf
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import csv
import sys
from joblib import Parallel, delayed
import multiprocessing
from scipy.signal import find_peaks


def releseslope(lifetime, t_sed=1.6e3, p_m=3e3, p_min=1e-3):
    x = np.log(lifetime) - np.log(t_sed)
    y = np.log(p_m) - np.log(p_min)
    return y / x

def snrlife(slope, t_sed=1.6e3, p_m=3e3, p_min=1e-3):
    dx = (np.log(p_m) - np.log(p_min)) / slope
    return np.exp(dx) * t_sed

start = datetime.now()
N = 45
Ea = np.logspace(-3,np.log10(3e3),200)
ages = np.logspace(5,7,N)
slope = np.linspace(1,2,N)
excesses = np.linspace(0,0.01,90)
lifetimes = snrlife(slope)
center = np.loadtxt('H4000_R20_D0norm')
left = np.loadtxt('H4000_R20_D0norm_Ratehalf')
right = np.loadtxt('H4000_R20_D0norm_Ratedouble')
prob = np.loadtxt('H4000_R20_D0norm_sedovdouble')
probmaxdist = np.loadtxt('maxdist')
debugprob = np.load('numb_sources_300pc_10xtime_1e5realizationalt.npy')*10
approx_DAMPE = 1.983*1.58308693**np.arange(9)
greenssum1 = np.load('greens_sum_point_bounded.npy')
greenssumuc = np.load('greens_sum_point_uncredit.npy')
greenssum3 = np.load('greens_sum_point_boundedup.npy')
greenssum2 = np.load('/home/uq036563/excess/sumgreenspoint.npy')*(approx_DAMPE)**2.2
greenssum = np.abs((greenssum1[1:]-greenssum3[:-1])/(greenssum1[1:]+greenssum3[:-1]))
Antons = np.load('Anton_point_stat_10TV_100ykr_fiducial.npy')
greensint = np.load('green_sum_integrated.npy')
interror = np.load('green_sum_integrated_error.npy')
greensdif = np.abs((-greenssum1+greenssumuc)/(greenssum1+greenssumuc))
#print(greensdif)
#peaks, _ = find_peaks(life[1]/(N+10-np.arange(N)))
# plt.plot(ages,one10kyr,label='1000pc/base')
# #plt.plot(ages,onetest/one1myr,label='base/1myr')
# #plt.plot(ages,one100kyr/onetest,label='100kyr/base')
# #plt.plot(ages[peaks],life[1][peaks]/life[0][peaks], 'x')
# plt.xscale('log')
# plt.xlabel('age/year')
# plt.title('ratio of probability of seeing peak')
# plt.legend()
# plt.savefig('accepted.png')
#peaklife = slope[peaks]
#print(np.diff(peaklife))
maxage = 2e5
sigmas =  0.2 * np.arange(26)
Ju = np.linspace(0,0.01,21)
breakE = np.logspace(-2.9,3.5,N)
ages = np.logspace(4.5,6.3,50)
distances = np.linspace(100,10000,90)
power = 3*maxage/100*(2000**2/2e4**2)
plot = probmaxdist[:,:26]*100
norm = colors.Normalize(np.min(greenssumuc[:,-5:]),np.max(greenssumuc[:,-5:]))
maxdists = np.array([100,200,300,400,500,600,1000,2000,4000])/1000
binnumber = 9 * (np.arange(45) + 1)
maxages = np.logspace(np.log10(1e5/4),np.log10(4*1e5),9)/1000
#ax1 = sb.heatmap(sysall.T,xticklabels=sigmas,yticklabels=["%.1g" % x for x in breakE][::3])
#ax1.yaxis.set_major_locator(plt.MaxNLocator(16))
#plt.pcolormesh(sigmas,np.arange(15)+1,debugprob[:,:26],norm=norm)
#plt.pcolormesh(sigmas[:23],np.arange(15)+1,debugprob[0,:23]/debugprob[:,:23])
norm1 = colors.LogNorm(np.min(greenssumuc[:100]),np.max(greenssumuc[:100]))
plt.pcolormesh(np.arange(9),4*np.arange(1000)[:100],greenssumuc[:100],norm=norm1)
#plt.vlines(2.13,maxdists[0],maxdists[-1],color='red',linewidth=0.5)
#plt.imshow(sysall.T,interpolation='bilinear')
#plt.plot(np.linspace(0,5,len(Antons)),Antons)
#plt.xlabel('$\sigma_{stat}$')
plt.xlabel('bin number')
#plt.xscale('log')
#plt.yscale('log')
plt.ylabel('distances')
#fig = ax1.get_figure()
plt.title('raitio of probabilities')
plt.colorbar(label='ratio of probabilities')
#cont = plt.contour(sigmas,maxdists,plot,levels=[30,84.1,93.3,97.7,99.3,99.9],colors='b',linewidths=0.3)
#plt.clabel(cont,fontsize=5.5,colors='k',alpha=1)
#plt.xlim(0,5)
#plt.legend(['all','1-200GeV','200-1000GeV','1-3000TeV'])
plt.savefig('greens_sum_point_uncredit2.png',dpi=150)

# plt.rc('font', size=22)
# fig, axs = plt.subplots(1,3,sharey=True,figsize=(30,10))

# plotleft = left[:,26:]*100
# plotcenter = center[:,26:]*100
# plotright = right[:,26:]*100
# cb = axs[0].pcolormesh(sigmas,breakE,plotleft,norm=norm,shading='flat')
# axs[0].title.set_text('rate halved')
# cont0 = axs[0].contour(sigmas,breakE,plotleft,levels=[30,84.1,93.3,97.7,99.3,99.9],colors='b',linewidths=1)
# axs[0].clabel(cont0,fontsize='smaller',inline=0,colors='k',alpha=1,fmt='%1.1f')

# axs[1].pcolormesh(sigmas,breakE,plotcenter,norm=norm,shading='flat')
# axs[1].title.set_text('Fiducial Values')
# cont1 = axs[1].contour(sigmas,breakE,plotcenter,levels=[30,84.1,93.3,97.7,99.3,99.9],colors='b',linewidths=1)
# axs[1].clabel(cont1,fontsize='smaller',inline=0,colors='k',alpha=1,fmt='%1.1f')

# axs[2].pcolormesh(sigmas,breakE,plotright,norm=norm,shading='flat')
# axs[2].title.set_text('rate doubled')
# cont2 = axs[2].contour(sigmas,breakE,plotright,levels=[30,84.1,93.3,97.7,99.3,99.9],colors='b',linewidths=1)
# axs[2].clabel(cont2,fontsize='smaller',inline=0,colors='k',alpha=1,fmt='%1.1f')

# for ax in axs.flat:
#     ax.set_yscale('log')
#     ax.set_ylim(10**(-2.9),10**(3.5))

# axs[0].set_ylabel('break Rigidity [TV]')
# axs[0].set_xlabel('$\sigma_{stat}$')
# axs[1].set_xlabel('$\sigma_{stat}$')
# axs[2].set_xlabel('$\sigma_{stat}$')
# cax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
# fig.colorbar(cb, label='probability of seeing at least a jump [%]',ax=axs[:],location='bottom')
# fig.tight_layout()
# plt.savefig('changerate.png',doi=1500,bbox_inches='tight')