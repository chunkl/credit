import scipy.integrate as integrator
from scipy.special import gamma, gammaincc
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from astropy.io import fits
import healpy as hp
import os
#from multiprocessing import Pool

'''
This Code calculates a skymap for unresolved sources, following the Vecchiotti et. al. 2022 ( https://iopscience.iop.org/article/10.3847/1538-4357/ac4df4 ), 
for the TeV energy range. Therefore we used the spatial source distribution following Lorimer et. al. 2006.
We have the appility to use the symetry of the problem and therefore it is possible to calculate only one quadrant of the skymap and the use the symetry to claculate the rest.
This procedure has the huge advantage to speed up the calculation a lot.
This code use a few Parameter which should be choosen before:
    Num_Core: Intiger; gives the number of Cores used in the Calculation, here we used the Pool-method from the multiprocessing-package
    full_skymap_calculation: boolen; if True, for each pixel in the full skymap the integrla will be calculated,
                                     if Flase, we use the symetry method and calculate only a part which speeds up the calculation
    plot: boolen; if True, the skyamp is plotted, if False the skymap is not plotted
'''

Num_Core = 1
full_skymap_calculation = False
plot = True
masking = True
lon_max = 100
lon_min = 25
lat_max = 5
lat_min = -5

'''Set some map parameter and calculate the angle for each pixel of the skymap in radiant and transfom it in from healpy in galactic coordinates'''
nside = 16
npix  = hp.nside2npix(nside)
pix = np.linspace(0, npix-1, npix, True, dtype=int)
b,l = hp.pix2ang(nside, pix)
b = -(b - np.pi/2) #Transform the Lattidue to a Coordinatesystem, where b is definde as the angle from the galactic centre towards the pole

'''Set some paramerter:'''
r_sun    = 8.5  #kpc ; 8.3kpc(CRINGE) oder 8.5kpc(Lorimer paper)?
alpha    = 1.5
R        = 1.9 * 10**-2 # yr⁻¹ Birth rate of PWNe 
tau      = 1.8 * 10**3  # yr   Lifetime of PWNe ???
L_max    = 3.058 * 10**35 #TeV/s        # 4.9*10**35 # erg/s
L_min    = 1.e-3 * L_max    # 3.02*10**35
beta     = 2.3
Ecut     = 500  #TeV
phi_CRAB = 2.26*10**(-11) #cm⁻² s⁻¹ 
phi_th   = np.sqrt(0.1*0.01) * phi_CRAB #geometric mean between 0.1 and 0.01 Crab nebula flux
H        = 0.5  #kpc
R_max    = 15 #kpc
kpc2cm = 3.086*10**21 #

"""
Lorimer parameter: 
    Here we set the parameter for the spatial source distribution, following Lorimer et. al. 2006 (LINK!!!!!!!!!!!!)
    we will follow the so called C-model
    If Lorimer_plot is True we will also plot the source distribution of both models
    If total_unmber is True we will calculate the total number of sources, this will have some deviations to the number of sources to the paper depending on the point, that we use another solar position
"""
Lorimer_plot = False # if True 
total_number = True#
model = 'S'             # string ('C' or'S' ) we can choose between the two models (C,S) described in Lorimer et. al.
if model == 'C':
    A = 41  #kpc⁻² 
    B = 1.9
    C = 5

elif model == 'S':
    A = 44 #kpc⁻²
    B = 0.2
    C = 1.4
else:
    sys.exit("Variable model is not S or C")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Defining source distribution functions from Lorimer et al. 2006 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Lorimer_rad( r , A=A, B=B, C=C, r_sun=r_sun):
    '''
    Input:
        r: radial distance to the galactic centre
        A, B, C are model parameter from Lorimer et al
        r_sun: position of the sun in kpc 
    Output:
        source distribution from Lorimer et. al., 2006, only the radial contribution 
        unit kpc⁻²
    '''
    return A * ( r / r_sun )**B * np.exp( - C * ( r - r_sun ) / r_sun )

def Lorimer_tot( r, z, H=H, A=A, B=B, C=C, r_sun=r_sun):
    '''
    Input:
        r: radial distance to the galactic centre
        z: distance perpendicular to the galactic plane
        H: Halo height default set by parameter above in this code
        A, B, C are model parameter from Lorimer et al
        r_sun: position of the sun in kpc 
    Output:
        Source distriubution from Lorimer et al include the exponential smearing in thr z-direction
        unit kpc⁻³
    '''
    return Lorimer_rad( r, A, B, C, r_sun) * np.exp( - z/H ) / (2*H)


def Lorimer_int(r, A=A, B=B, C=C, r_sun=8.5):
    '''
    Input:
    Output:
    Note:
        This function is only used for the calculation of the total number, for better comparision wwith the Lorimer papaer set the r_sun parameter to 8.5 kpc 
    '''
    return 2*np.pi*r*Lorimer_rad(r, A=A, B=B, C=C, r_sun=r_sun)

def roh( r, z, A=A, B=B, C=C, H=H, r_sun=r_sun, R_max=R_max):
    roh_R = A * pow(r/r_sun, B) * np.exp( - C * ( r - r_sun )/r_sun)
    N_z = np.exp(-abs(z)/H)
    norm_R = A * pow(C,-2-B) * np.exp(C) * np.pi * pow(r_sun,2) * gamma(2+B) * ( 1 - gammaincc(2+B, C*R_max/r_sun)) # Normalise the radial debendencie to 1 in the range up to r_max 
    norm_z = 2 * H
    return roh_R * N_z /(norm_R * norm_z)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Defining coordinate transformation: r(s,l,b) and z(s,b)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def r( s, l, b, r_sun=r_sun ):
    '''
    Input
        s: distance from our position, units kpc
        l: galactic longitude in rad
        b: galactic latitude in rad
    Output: 
        r: radial distance to the galactic centre, units kpc (same as s)
    '''
    return np.sqrt( s**2 * np.cos(b)**2 + r_sun**2 - 2 * r_sun * s * np.cos(l) * np.cos(b) )

def z(s,b):
    '''
    Input
        s: distance from our position
        l: galactic longitude in rad
        b: galactic latitude in rad
    Output: 
        z: distance perpendicular to the galctic plane
    '''
    return np.abs( s * np.sin(b) )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Defining functions from Vecchiotti et. al. 2022:
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Y(L, L_max=L_max, alpha=alpha):
    return (alpha-1)/L_max * (L/L_max)**(-alpha)

def Gamma (L, beta=beta):
    return (beta - 2)/(beta - 1) * L * ( 1 - 100**(2 - beta) )/ ( 1 - 100**(1 - beta) )**-1 #Check the last term ###This last term makes the skymap a factor 0.866 smaler

def phi(E, Ecut=Ecut, beta=beta ):
    '''
    Input:
        E: float or array like object; Energy in TeV
        E_cut: float; Energy in TeV, default=500TeV
        beta: float; spectral index for this energy range, default=2.3
    Output: 
        float or arry-like object, depending on input, result of the function phi(E) in TeV
    To Do:
        Calculate the energy dendence ( function phi(E) ), following Vecchiotti et. al. 2022 ( https://iopscience.iop.org/article/10.3847/1538-4357/ac4df4 ), which is similar to the CRINGE-paper
    '''
    return (beta - 1)/( 1 - 100**(1-beta)) * (E)**(-beta) * np.exp(- E/Ecut )

def D(L, phi_th=phi_th):
    return np.sqrt( Gamma(L)/(4 * np.pi * phi_th) )/kpc2cm 


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Defining the integrands 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def LoS_integrand(s, l, b ):
    return Lorimer_tot( r(s, l, b), z(s,b) ) #* 2*np.pi * r(s, l, b) 

def integrand( s, L, l, b):
    return Y(L) * Gamma(L) * LoS_integrand(s, l, b )

def Integration(i):
    return integrator.dblquad(integrand, L_min, L_max, D, 30, (l[i],b[i]))[0]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Here we calculate the integration via nquad 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def integrand_nquad( L, s, l, b):
    return Y(L) * Gamma(L) * LoS_integrand(s, l, b )

def bounds_L():
    return [L_min,L_max]

def bounds_s(L):
    return [D(L), 30]

def Integration_nquad(i):
    return integrator.dblquad(lambda s, L: Y(L) * Gamma(L) * roh ( r(s,l[i],b[i]), z(s,b[i]) ), L_min, L_max, lambda L: D(L), lambda L:30)[0]
    # return integrator.nquad(integrand_nquad,ranges=[ bounds_s, bounds_L ], args=(l[i],b[i]))[0]


def masking( map, map_shape,lon_range, lat_range, nside=nside, lonlat=True, dim=True):
    '''
    Input:
        map: array, holds a map, which should be masked
        lon_range: tuple in the form (lon_minimum, lon_maximum) in degrees if lonlat is True; describes the longitude range, inside will be hold the data outside will be masked
        lat_range: tuple in the form (lat_minimum, lat_maximum) in degrees if lonlat is True; same as above
        nside: the nside value of the map
        lonlat: boolen sets the unit to degree if True, to radiant if False; default value is True
        1dim: boolen default is True; if True a pure skymap is given with shape (12*nside²), if False a 2dim array is given with shape (x, 12*nside²), where X is an int
    Output:
        masked map
    ToDo:
        masking an given map 
    '''
    lon, lat = hp.pix2ang(nside, np.arange(12*nside**2), lonlat=lonlat)

    if lon_range[0]>=0:
        index = np.sort(np.append(np.where( lon<lon_range[0] ), np.where( lon>lon_range[1] ) ))
    else:
        index = np.sort( np.where( (lon > lon_range[1]) & (lon < 360 + lon_range[0] ) ))              #Note: lon_range[0] is here smaller 0

    index = np.append(index, np.where( lat<lat_range[0] ) )
    index = np.sort( np.unique( np.append(index, np.where( lat>lat_range[1] ) )) )

    mask = np.full( map_shape ,False, dtype=bool)
    
    if dim:
        for i in index:
            mask[i] = True
    else:
        for j in range(map_shape[0]):
            for i in index:
                mask[j,i] = True

    masked_map = np.ma.array( map, mask=mask, fill_value= hp.UNSEEN, )
    return masked_map


if __name__ == '__main__':
    print('Starting at: {}'.format(time.ctime(time.time())))
    t0 = time.time()

    if total_number:
        print('Integral of the radial distribution from 8 to 9 kpc: ', integrator.quad(Lorimer_int, 8,9 )[0])
        print('Integral of the radial distribution from 0 to 15 kpc: ', integrator.quad(Lorimer_int, 0, 15 )[0])
        print('Integral of the radial distribution from 0 to 20 kpc: ', integrator.quad(Lorimer_int, 0, 20 )[0])
        print('Integral of the radial distribution from 0 to 25 kpc: ', integrator.quad(Lorimer_int, 0, 25 )[0])
        print('Integral of the radial distribution from 0 to inf kpc: ', integrator.quad(Lorimer_int, 0, np.inf )[0])

    if full_skymap_calculation:
        # Claculate the skymap:
        print('Starting to calculate the full skymap at: {}'.format(time.ctime(time.time())))
        t1 = time.time()

        with Pool(Num_Core) as p:
            skymap = p.map(Integration_nquad, pix)
        skymap = R * tau * skymap/(4*np.pi*kpc2cm**2) /npix     # Claculate kpc⁻² into cm⁻² and multiply with other factors ignored above
        
        np.save('unresolved_sources_nside{:.0f}.npy'.format(nside), skymap)
        print('Finishing the skymap calculation at: {}'.format(time.ctime(time.time())))
        print('Runtime of the skymap calculation was: {}s'.format( time.time() - t1 ))

        if plot:
            # Plot the skymap:
            hp.mollview( np.log10(skymap) ,cmap='jet', title='Unresolved sources test')
            filename ='unresolved_sources_full_sky_calculation_nside{:.0f}'.format(nside)

            hp.graticule(dpar=10,dmer=10)
            plt.savefig('Skymap_' + filename + '.png')
    

    ##################
    else:
        #masking:
        mask_quad_1 = (b>=0) & (l<=np.pi)
        pix_quad_1  = pix[mask_quad_1]
        b_quad_1    = b[mask_quad_1]
        l_quad_1    = l[mask_quad_1]

        # Claculate the skymap:
        print('Starting to calculate the first quadrant of the skymap at: {}'.format(time.ctime(time.time())))
        t1 = time.time()

        # with Pool(Num_Core) as p:
        #     X = p.map(Integration, pix_quad_1)
        for i in pix_quad_1:
            X = Integration_nquad(i)
        print(time.time()-t1)
        skymap = np.zeros(npix)
        skymap[pix_quad_1] = X
        skymap = R * tau * skymap/(4*np.pi*kpc2cm**2) # * npix)    # Claculate kpc⁻² into cm⁻² and multiply with other factors ignored above
        np.save('unresolved_sources_quadrant_1_nside{:.0f}.npy'.format(nside), skymap)
        print( 'The first quadrant of the skymap is saved in: "unresolved_sources_quadrant_1_nside{:.0f}.npy"'.format(nside) )

        #skymap = np.load('unresolved_sources_quadrant_1_nside{:.0f}.npy'.format(nside))
        
        print('Finishing the calculation of the 1st quadrant of the skymap at: {}'.format(time.ctime(time.time())))
        print('Runtime of the skymap calculation was: {}s'.format( time.time() - t1 ))


        # Calculate the whole skymap:       
        for i in np.unique(l_quad_1):
            # print(i)
            # print(pix[(b<0)& (l==i)])
            # print(np.flip(pix[(b>0) & (l==i)]))
            skymap[(b<0)& (l==i)] = np.flip( skymap[(b>0) & (l==i)])
        
        for j in np.unique(b):
            # print(j)
            # print(pix[(b==j) & (l<np.pi) & (l>0) ])
            # print(np.flip(pix[(b==j) & (l>np.pi)]) )
            skymap[ (b==j) & (l>np.pi) ] = np.flip( skymap[ (b==j) & (l<np.pi) & (l>0) ])


        np.save('unresolved_sources_full_sky_nside{:.0f}.npy'.format(nside), skymap)
        print( 'The full skymap is saved in: "unresolved_sources_full_sky_nside{:.0f}.npy"'.format(nside) )

        if plot:
            # Plot the skymap:
            hp.mollview( np.log10(skymap) ,cmap='jet', title='Unresolved sources test')
            filename ='unresolved_sources_quadrant_1_2_3_4_nside{:.0f}'.format(nside)

            hp.graticule(dpar=5,dmer=5)
            plt.savefig('Skymap_' + filename + '.png')

    if masking:
        # Emin = 10
        # Emax = 10**8
        # nE   = 50  
        unresolved_sky = skymap #*10**-4 # willkürlich
        print(unresolved_sky.shape)
        # unresolved_flux_total_sky = phi(energy * 10**(-3) ) * 10**(3) * unresolved_sky
        # unresolved_flux_masked = masking(unresolved_flux_total_sky, unresolved_flux_total_sky.shape, (lon_min, lon_max), (lat_min, lat_max), 128 )
        unresolved_masked_map = masking(unresolved_sky, unresolved_sky.shape, (lon_min, lon_max), (lat_min, lat_max), nside )
        print(unresolved_masked_map.sum(axis=0)/unresolved_masked_map.count(axis=0) ) #*4*np.pi ) #####################################################################4*np.pi insert
        print('the result should be ~1.16e-10')
        print(unresolved_masked_map.count(axis=0))
        #print('fraction: {:.3e}'.format( (unresolved_masked_map.sum(axis=0)/unresolved_masked_map.count(axis=0))/(1.16*10**-10) ) )
        hp.mollview(unresolved_masked_map)
        hp.graticule(dpar=10,dmer=10)
        plt.savefig('unresolved_masked_test.png')
        # energy = np.logspace( np.log10(Emin), np.log10(Emax), nE, True) #unit GeV

        # unresolved_flux_sky_window = unresolved_masked_map.sum(axis=0)/unresolved_masked_map.count(axis=0) * phi(energy * 10**(-3) ) * 10**(-3) #* ( 1 - 100**(-0.3) )
        # print(unresolved_flux_sky_window)

        





    # print(l*180/np.pi)
    # print(b*180/np.pi)
    # print(z(1,b))
    # print( -(b - np.pi/2)*180/np.pi)
    # print( z(1, -(b - np.pi/2)))

    # print(integrator.dblquad(integrant, L_min, L_max, D, np.inf, (0,0) ))

    # print(r(0,0,0), r(-8.3,np.pi,0),r(1,0,0))

    # print(integrator.quad(LoS_integrand, -8.3, 11.7, (np.pi,0) )[0])


    if total_number:
        print('Integral of the radial distribution from 8 to 9 kpc: ', integrator.quad(Lorimer_int, 0, 15 )[0])
        print('Integral of the radial distribution from 0 to 15 kpc: ', integrator.quad(Lorimer_int, 0, 15 )[0])
        print('Integral of the radial distribution from 0 to 20 kpc: ', integrator.quad(Lorimer_int, 0, 20 )[0])
        print('Integral of the radial distribution from 0 to 25 kpc: ', integrator.quad(Lorimer_int, 0, 25 )[0])
        print('Integral of the radial distribution from 0 to inf kpc: ', integrator.quad(Lorimer_int, 0, np.inf )[0])


    if Lorimer_plot:
        r = np.linspace(0, 15, 60) 
        if model == 'C':
            A = 41 / (3.241*10**22)**2 # cm⁻²
            B = 1.9
            C = 5
            plt.plot( r, Lorimer_rad(r*kpc2cm,A,B,C) / Lorimer_rad(r_sun,A,B,C), label='model C')

        elif model == 'S':
            A = 44 / (3.241*10**22)**2 # cm⁻²
            B = 0.2
            C = 1.4
            plt.plot( r, Lorimer_rad(r*kpc2cm ,A,B,C)/Lorimer_rad(r_sun*kpc2cm,A,B,C), label='model S')

        plt.grid()
        plt.legend()
        plt.ylabel('S(r)/S(r=r_sun)')
        plt.xlabel('r in kpc')
        plt.savefig('Lorimer.png')

    print('Runtime: {}s'.format(time.time()-t0) )
    print('End at: {}'.format(time.ctime(time.time())))