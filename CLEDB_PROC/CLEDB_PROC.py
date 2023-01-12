# -*- coding: utf-8 -*-

###############################################################################
### Parallel implementation of utilities for CLEDB_PROC                     ###
###############################################################################

#Contact: Alin Paraschiv, High Altitude Observatory
#         arparaschiv@ucar.edu
#         paraschiv.alinrazvan+cledb@gmail.com

### Tested Requirements  ################################################
### CLEDB database as configured in CLE V2.0.3; (db203 executable) 
# python                    3.10.8 
# scipy                     1.9.3
# numpy                     1.23.4 
# numba                     0.56.4 
# llvmlite                  0.39.1 (as numba dependency probably does not need separate installing)


### Auxiliary, only used in verbose mode
# time
# os
# sys

### Crutches for data handling during testing, not utilized by the inversion
# importlib ## to recompile the auxiliary file before each run
# pickle    ## convenient python/numpy data storage and loading
###


### NUMBA NOTES: ##############################################################
## Generally, if a numba non-python  (njit, or jit without explicit object mode flag) 
## calls another function, the second should also be non-python compatible 

## Array indexing by lists and advanced indexing are generally not supported by numba.
## Do not be fooled by the np. calls. All np. functions used are rewritten by numba.
## The implementation reverts to simple slicing, and array initialization by tuples.

## Disk IO operations with the sys and os modules are not numba non-python compatible. 
## They do work in object-mode and loop-lifting is enabled. Enabling numba does make sense.

## global variables can't really be used with numba parallel python functions unless you aim to keep everything constant.

## Functions in the time module are not understood/supported by numba. enabling them forces the compiler to go back to object mode and throw warnings. 
## Full Parallelization will generally not be possible while time functions are enabled.



#########################################################################
### Needed imports ######################################################
#########################################################################

#
# Needed libraries
#
import numpy as np
from numba import jit,njit,prange
import scipy.stats as sps
from scipy.optimize import curve_fit
import time
import constants
import ctrlparams 
params=ctrlparams.ctrlparams()    ## just a shorter label

## optional libraries
# import time
# from pylab import *
# import glob
# import os
# import sys
# import numexpr as ne

###########################################################################
### Main Modules ##########################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,cache=params.jitcache)
def cledb_invproc(sobs_totrot,sobs_dopp,database,db_enc,yobs,aobs,rms,dbhdr,keyvals,nsearch,maxchisq,bcalc,iqud,reduced,verbose):

    if verbose >=1: print('--------------------------------------\nCLEDB_IVPROC: TWO LINE INVERSION START\n--------------------------------------')
    
    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny = keyvals[0:2]
    
    ##Output variables
    invout = np.zeros((nx,ny,nsearch,11),dtype=np.float32)
    sfound = np.zeros((nx,ny,nsearch,8),dtype=np.float32)
    
    ## Number of degrees of freedom >4; we require a two line full IQUV observation.
    ## assume the last dimension of sobs is the number of observables; 2 lines=8
    ## Database needs to match the observation
    if sobs_totrot.shape[-1] != 8 or database[0].shape[-1] != 8: 
        if verbose >=1:
            print('CLEDB_INVPROC: Must have a two line observation and/or a corresponding two line database; Aborting!')
            print("Shapes are: ", sobs_totrot.shape," and ",database[0].shape)
        invout[:,:,:,0]=np.full((nx,ny,nsearch),-1) ### index is -1 for failed runs!
        return invout,sfound
    
    ### NO wave observation and matching for IQUD --> NO RUN  ######
    if (iqud == True and bcalc != 3) or (iqud != True and bcalc == 3):
        invout[:,:,:,0]=np.full((nx,ny,nsearch),-1) ### index is -1 for failed runs!
        if verbose >= 1: 
            print("WARNING (CLEDB_MATCH): Field strength calculation incompatible with requested input data! Check bcalc and iqud keywords")
        return invout,sfound
    
    ## We use the iqud keyword to run the matching function.
    if iqud == True:
        if verbose >= 1:      
            for xx in range(nx): #range(301,303): #
                print("CLEDB_INVPROC (IQUD): Executing ext. loop:",xx," of ",nx," (",ny," calculations / loop)")
                for yy in prange(ny): #prange(100,200): #
                    invout[xx,yy,:,:],sfound[xx,yy,:,:]=cledb_matchiqud(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],\
                                                                        nsearch,maxchisq,bcalc,reduced,verbose)
        else:
            for xx in range(nx):
                for yy in prange(ny):
                    invout[xx,yy,:,:],sfound[xx,yy,:,:]=cledb_matchiqud(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],\
                                                                    nsearch,maxchisq,bcalc,reduced,verbose)
    else:
        if verbose >= 1:      
            for xx in range(nx): #range(301,303): #
                print("CLEDB_INVPROC (IQUV): Executing ext. loop:",xx," of ",nx," (",ny," calculations / loop)")
                for yy in prange(ny): #prange(100,200): #
                    # if xx == 301 and yy == 105:
                    #     sobs_totrot[xx,yy,:]=np.array((1.00000000e+00, 3.16772669e-04, 1.52991570e-03, 5.63795435e-06, 6.24271335e-01, 1.66090059e-05, 8.02164495e-05, 3.53448432e-06),dtype=np.float32) ##5,25,52,70 or 5269750
                    #     print(database[db_enc[xx,yy]][5,25,52,70,:])
                    #     print(sfound[xx,yy,:,:]/sfound[xx,yy,:,0])
                    #     print(invout[xx,yy,:,0:2])
                    # else:
                    #     invout[xx,yy,:,:],sfound[xx,yy,:,:]=cledb_matchiquv(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],\
                    #                                                     nsearch,maxchisq,bcalc,iqud,reduced,verbose)             
                    invout[xx,yy,:,:],sfound[xx,yy,:,:]=cledb_matchiquv(sobs_totrot[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],\
                                                                        nsearch,maxchisq,bcalc,reduced,verbose)
        else:
            for xx in range(nx):
                for yy in prange(ny):
                    invout[xx,yy,:,:],sfound[xx,yy,:,:]=cledb_matchiquv(sobs_totrot[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],\
                                                                    nsearch,maxchisq,bcalc,reduced,verbose)


    if verbose >=1:
        #if params.verbose >=3: print("{:4.6f}".format(time.time()-start),' TOTAL SECONDS FOR DBINVERT ')
        print('--------------------------------------\nCLEDB_INVPROC: 2 LINE INVERSION FINALIZED\n--------------------------------------')

    ######################################################################
    ## [placeholder for issuemask]

    return invout,sfound
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)
def blos_proc(sobs_tot,rms,keyvals,consts,params):
## Uses the "improved" magnetograph formulation in eq 40 Casini &Judge 99, eq 14 of Plowman 2014, and eq 17 and 18 of Dima & Schad 2020
## Follows the discussion in the three papers and adopts different analythical implementations based on the line combination used
## As shown in the papers the magnetograph formulation is not precise in terms of recovering the LOS magnetic field.
## differences of the order of $\pm$ 2 times actual values  based on the atomic alignment, F factor, and LOS angle $\theta$ can manifest in that product.

## script will produce one set of 2 times degenerate magnetograph along with a classic magnetograph and a field azimuth for each line/observation that is fed via sobs_tot and keyvals.

    if params.verbose >=1: print('--------------------------------------\n---BLOS_PROC: LOS B ESTIMATION START--\n--------------------------------------')
    ## unpack needed values from keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny = keyvals[0:2]
    tline = keyvals[4] 
    ## needed array initialization
    blosout = np.zeros((nx,ny,4,len(tline)),dtype=np.float32)
    
    for zz in range(0,len(tline)):
        ## initialize constants
        const=consts.Constants(tline[zz])                       ## for each line load the correct constants

        ## the list of the F factors from Dima & Schad 2020 for lines of interest are saved in the constants class; 
        ## both Fe XIII and SI IX have F == 0.
        if params.verbose >=2 and tline[zz] == "si-x_1430":
            print('BLOS_PROC: Constrained magnetograph solutions will have an additional correction applied.')

        ## two degenerate solutions are produced for blos; Indexes 0 and 1
        ## the accurate solution matches sign with \sigma ^2_0 atomic alignment.
        ## the "standard" magnetograph formulation (index 2) represents, in practice, the average accuracy between the two degenerate solutions.
        ## Note: this solution is inaccurate in almost all cases; with the exception of fields tangential to the radial direction.

        for xx in range(nx):
            for yy in prange(ny):
                ## linear degree of polarization total
                lpol = np.sqrt(sobs_tot[xx,yy,1+(4*zz)]**2 + sobs_tot[xx,yy,2+(4*zz)]**2)
                ## compute the azimuthal component (phi) of the magnetic field from linear polarization
                blosout[xx,yy,3,zz] = - 0.5 * np.arctan2(sobs_tot[xx,yy,2+(4*zz)],sobs_tot[xx,yy,1+(4*zz)])
                if (sobs_tot[xx,yy,0+(4*zz)] != 0) and (np.isnan(sobs_tot[xx,yy,4*zz:4*(zz+1)]).any() == False):          ### captures for nans or division by 0
                    ## for corrected magnetograph magnetic field implement eq.17 from Dima & Schad 2020
                    ## sin^Theta_B ==1; theta_b=90 cf chap 4.3 Dima & Schad 2020 to minimize deviation from "true" solution.
                    ## 1e9 converts SI constants to nm to divide by the reference wavelength also in nm
                    #blosout[xx,yy,0,zz]=(1/(4.67e-12*(const.line_ref**2)) ) * ( -sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*(sobs_tot[xx,yy,0+(4*zz)] - lpol) - 0.66*const.F_factor*(lpol/1.)) )
                    blosout[xx,yy,0,zz]=(const.planckconst/const.bohrmagneton*const.l_speed*1.e9/(const.line_ref**2) ) * ( -sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*(sobs_tot[xx,yy,0+(4*zz)] - lpol) - 0.66*const.F_factor*(lpol/1.)) )
                    blosout[xx,yy,1,zz]=(const.planckconst/const.bohrmagneton*const.l_speed*1.e9/(const.line_ref**2) ) * ( -sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*(sobs_tot[xx,yy,0+(4*zz)] + lpol) + 0.66*const.F_factor*(lpol/1.)) )
                    ## standard magnetograph formulation
                    blosout[xx,yy,2,zz]=(const.planckconst/const.bohrmagneton*const.l_speed*1.e9/(const.line_ref**2) ) * ( -sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*sobs_tot[xx,yy,0+(4*zz)]) )
                #else:
                #    blosout[xx,yy,0:3,zz]==np.nan  ### else branch not needed. blosout cast to empty is already nan in pixels where field can't be computed

    ######################################################################
    ## [placeholder for issuemask]

    if params.verbose >=1: print('--------------------------------------\n---BLOS_PROC: LOS B ESTIMATION END----\n--------------------------------------')   
    return blosout
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)
def spectro_proc(sobs_in,sobs_tot,rms,background,keyvals,consts,params):
## calculates 12 spectroscopic products
## Background is used both as an output product and as a preprocess parameter. More details in cdf_statistics comments.

    if params.verbose >=1: 
        print('--------------------------------------\nSPECTRO_PROC: SPECTROSCOPY START\n--------------------------------------')
        if params.verbose >= 3:start0=time.time()  
    
    start0=time.time()  
    ## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny,nw = keyvals[0:3]
    nline = keyvals[3]
    tline = keyvals[4]
    crpix3 = keyvals[7]                                              ## The wavelength domains will be different for each input line. nl keeps the index of the line to solve.
    crval3 = keyvals[10]
    cdelt3 = keyvals[13]
    
    ## needed data arrays
    specout=np.zeros((nx,ny,2,12),dtype=np.float32)                  ## output array containing the spectroscopic products.
    wlarr=np.zeros((nw,nline),dtype=np.float32)                      ## wavelength array
    sobs_cal=np.zeros((nx,ny,nw,nline*4),dtype=np.float32)           ## define this as 0 to be able to later check if the calibs are performed or not.
    ##NOTE: sobs_cal output should be in the shape of an array; e.g. [x,y,w,4*nline]
    
    ## create a wavelength array based on keywords for each line to be processed
    for i in prange(nline):
        if crpix3[i] == 0:                                          ## simple; just cycle and update
            for j in range(nw):
                wlarr[j,i]=crval3[i]+(j*cdelt3[i])
        else:                                                       ## If reference pixel is not 0;  cycle for forwards and backwards from crpix3
            for j in range(crpix3[i],nw):                           ## forward cycle
                wlarr[j,i]=crval3[i]+((j-crpix3[i])*cdelt3[i])
            for j in range(crpix3[i],-1,-1):                        ## Backwards cycle; -1 to fill the 0 index of the array
                wlarr[j,i]=crval3[i]-((crpix3[i]-j)*cdelt3[i])

    ######################################################################
    ## placeholder for LEV2CALIB_WAVE
    ## level 2 absolute wavelength calibration using photospheric and telluric absorption profiles.
    ## to be implemented after LEV1 data corrections are known and detailed (if needed).

    #sobs_cal=lev2calib_wave(sobs_in,rms,params.verbose)

    ######################################################################
    ## placeholder for LEV2CALIB_ABSINT
    ## level 2 absolute intensity calibration, using center to limb variation and close-to-limb on-disk flux measurements. 
    ## to be implemented after LEV1 data corrections are known and detailed (if needed).

    #if np.count_nonzero(sobs_cal) != 0:                           ## Check if LEV2CALIB_WAVE is performed
    #    sobs_cal=lev2calib_wave(sobs_cal,rms,params.verbose)      ## just update the wavelength calibrated sobs
    #else:
    #    sobs_cal=lev2calib_wave(sobs_in,rms,params.verbose)       ## apply the intensity calibration without the wavelength calibration

    ######################################################################
    ## define sobs_cal if no LEV2CALIB_ABSINT or LEV2CALIB_WAVE is performed

    if np.count_nonzero(sobs_cal) == 0:                       ## Check if any of LEV2CALIB_WAVE and/or LEV2CALIB_ABSINT is performed
        if nline == 2:            
            sobs_cal=np.append(sobs_in[0],sobs_in[1],axis=3).reshape(nx,ny,nw,8)
        else:
            sobs_cal=sobs_in[0]
            
    ######################################################################
    ## process the spectroscopy                     
    if params.verbose >= 1: 
        if nline == 2:                  
            for xx in range(600,nx):
                print("SPECTRO_PROC: Executing ext. loop:",xx," of ",nx," (",ny," calculations / loop)")
                for yy in prange(500,ny):
                    specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                        background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)    
                    specout[xx,yy,1,:] = cdf_statistics(sobs_cal[xx,yy,:,4:8],sobs_tot[xx,yy,4:8],\
                        background[xx,yy,4:8],wlarr[:,1],keyvals,consts.Constants(tline[1]),params.gaussfit,params.verbose) 
                print("{:4.6f}".format(time.time()-start0),' oneiter')
        else:
            for xx in range(nx):
                print("SPECTRO_PROC: Executing ext. loop:",xx," of ",nx," (",ny," calculations / loop)")
                for yy in prange(ny):
                    specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                        background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)        
    else:
        if nline == 2:                  
            for xx in range(nx):
                for yy in prange(ny):
                    specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                        background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)    
                    specout[xx,yy,1,:] = cdf_statistics(sobs_cal[xx,yy,:,4:8],sobs_tot[xx,yy,4:8],\
                        background[xx,yy,4:8],wlarr[:,1],keyvals,consts.Constants(tline[1]),params.gaussfit,params.verbose)     
        else:
            for xx in range(nx):
                for yy in prange(ny):
                    specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                        background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)             
                
                ##Note: cdf_statistics will alter sobs_cal; It is not outputed or used downstream, so no issues should appear!

    ######################################################################
    ## placeholder for ML_LOSDISENTANGLE
    ## this should use machine learning techniques for population distributions to disentangle multiple "normal" emission profiles along the LOS.
    ## The goal is to provide an agnostic model that does not require manual input and judgement from the user; 
    ## e.g. multi-gausisan fit with n functions, where n is a manual judgement.

    ######################################################################
    ## [placeholder for issuemask]

    ## NOTE: SPECOUT will always have two dimensions at exit to keep data dimensionality consistent. 
    ##      In the case of just one line observations, the second dimension is not filled in.
    ##      The array can be reshaped outside of the numba enabled functions to drop the extra dimension if needed.
    if params.verbose >=1: 
        if params.verbose >= 3: print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL SPECTROSCOPY PROCESSING')
        print('--------------------------------------\nSPECTRO_PROC: SPECTROSCOPY FINALIZED\n--------------------------------------')
    return specout
###########################################################################
###########################################################################

###########################################################################
###########################################################################
### Helper functions ######################################################
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)
def cdf_statistics(sobs_cal,sobs_tot,background,wlarr_1pix,keyvals,const,gaussfit,verbose):
## compute the statistics of the Stokes I profiles using CDF methods and return results for 1 pixel.
## this is run for 1 ion (one set of IQUV). Outside loop controls which ion is fed.
## products are:
## line core wavelength;
## line shift from reference position;
## shift converted to velocities;
## Stokes I,Q,U, and V linecore intensity;
## background intensity of stokes I;
## Total line width;
## Non-thermal component of the line width;
## fraction of linear polarization;
## fraction of total polarization;
    start0=time.time()  
    #### NO OBSERVATION --> NO RUN! ###########    
    ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
    ##returns an array-like 0 vector of the dimensions of the requested output.
    if np.isnan(sobs_cal).all() or np.count_nonzero(sobs_cal) == 0:
        nullout = np.zeros((12),dtype=np.float32)
        if verbose >= 2: print("SPECTRO_PROC: FATAL! No observation in voxel!")
        return nullout
    
    ######################################################################    
    ## variable unpacks and preprocesses

    ## load what is needed from keyvals (these are unpacked so its clear what variables np.count_nonzero(sobs_cal) are being used. One can just use keyvals[x] inline.)
    nw = keyvals[2]
    cdelt3 = wlarr_1pix[1] - wlarr_1pix[0]                    ## computes cdelt3 locally;workaround to reduce the number of inputs variables; agnostic to which line it solves for
    instwidth = keyvals[15]

    # needed arrays
    spec_1pix=np.zeros((12),dtype=np.float32)                 ## output array containing the spectroscopic products.
    issuemask=np.zeros(4,dtype=np.int32)                      ## temporary placeholder for issuemask
 
    ######################################################################
    ## Preprocess the spectral data
    
    ## Noise reduce the spectral input data NOTE: convoluted process.  This is technically a preprocess step.
    ## background subtraction with Background and raw sobs_in arrays are processed here as part of cdf_statistics. 
    ## sobs_tot has background subtracted with the preprocess module via obs_integrate. BUT it is also normalized for the invproc and blosproc modules. So not optimal here.
    ## This is done this way because background is also desired as an output product, 
    
    #print(sobs_cal[:,0],background[0])
    #print(sobs_cal[:,0]-background[0])
    #for i in range(4):                                        ## range(4); only 1 line is fed at a time to cdf_statistics
    #    sobs_cal[:,i]=sobs_cal[:,i]-background[i]
    sobs_cal-=background
    print("{:4.6f}".format(time.time()-start0),' bkg')
    ## now recompute the cdf for the noise-reduced data
    cdf=obs_cdf(sobs_cal[:,0]) 

    ## check if there is reliable signal to fit and analyze
    ##(1) we can fit a line to the cdf distribution and in case it does fit well, there is no (reliable) stokes profile to recover.
    ##the 1.4e-5 rest in correlation corresponds to a gaussion with a peak in intensity of <5% compared to the average signal across the spectral range.
    print("{:4.6f}".format(time.time()-start0),' cfd')
    if 1.- sps.pearsonr(wlarr_1pix,cdf)[0] < 1.4e-5:
        if verbose >=3: print("SPECTRO_PROC: WARNING! No reliable Stokes I signal in pixel...")
        issuemask[0]=1
    print("{:4.6f}".format(time.time()-start0),' ps1')
    ## compute the center of the distribution, the line width, and the doppler shifts in wavelength and velocity.
    ## a normal distribution 0.5 is the average, sigma=34.13; FWHM=2*sqrt(2*alog(2))*sigma,
    ## [0] left stat fwhm, [1] center stat fwhm, [2] right stat fwhm;
    ## [3]left fwhm wave position, [4] fwhm center wave position, [5] fwhm right wave position;
    ## [6],[7],[8] just record the LEFT array indexes for recorded positions at [3], [4], [5];
    tmp=np.zeros((9),dtype=np.float32)                             ## use the tmp variable to store all the data;
    tmp[0:3]= 0.09815, 0.5, 0.90185  ## [0.5-2*np.sqrt(2*np.log(2))*34.13/2./100, 0.5, 0.5+2*np.sqrt(2*np.log(2))*34.13/2./100 ] ## /2/100 comes from half width --> percentage

    for j in range (0,3):
        for i in range (0,nw):   
            if cdf[i] <= tmp[j] < cdf[i+1]:                          ## find between which bins the distribution centre is.
                k1=2.*(tmp[j]-cdf[i])/(cdf[i+1]-cdf[i])               ## find the normed difference to theoretical centre from the left bin.
                tmp[j+3]=wlarr_1pix[i]+k1*cdelt3
                tmp[j+6]=i
            elif i == nw-1:
                if j == 1:
                    if verbose >=3: print("SPECTRO_PROC: WARNING! Line center not found")
                    issuemask[2]=1
                if j == 0:
                    if verbose >=3: print("SPECTRO_PROC: WARNING! FWHM left margin not found")
                    issuemask[3]+=1
                if j == 2:
                    if verbose >=3: print("SPECTRO_PROC: WARNING! FWHM right margin not found")
                    issuemask[3]+=1
    ## fudge for not doing repeated type conversions
    tmp6,tmp7,tmp8=np.int32(tmp[6:9])
    print("{:4.6f}".format(time.time()-start0),' for')
    ## check if there is reliable signal to fit and analyze  (2) if the distribution is skewed, the inner part of it should not fit a line.
    if tmp8 - tmp6 >= 2: 
        if 1. - sps.pearsonr(wlarr_1pix[tmp6:tmp8+1],cdf[tmp6:tmp8+1])[0] > 5e-3:
            if verbose >=3: print("SPECTRO_PROC: WARNING! Emission does not follow a normal distribution")
            issuemask[1]=1
    else:
    ## not reliable signal == no products! 
        nullout = np.zeros((12),dtype=np.float32)
        if verbose >= 2: print("SPECTRO_PROC: FATAL! Spectroscopy not resolvable in pixel")
        return nullout            
    print("{:4.6f}".format(time.time()-start0),' ps2')
    ######################################################################            
    ## The spectroscopic products for each line are computed here.   
    
    ## core wavelength and width
    if gaussfit == 2:                                                                  ## CDF + GAUSS version;
        gfit,gcov = curve_fit(obs_gaussfit,wlarr_1pix,sobs_cal[:,0],p0=[np.max(sobs_cal[:,0]),tmp[4],(tmp[5]-tmp[3])/2.3548,0.0],maxfev=5000)
        spec_1pix[0]=gfit[1]                                                           ## line core wavelength
        spec_1pix[8]=2.3548*gfit[2]                                                    ## Total line width [nm]; 2*np.sqrt(2*np.log(2))==2.3548
    elif gaussfit == 1:                                                                ## GAUSS version; still need cdf for extra calculations
        gfit,gcov = curve_fit(obs_gaussfit,wlarr_1pix,sobs_cal[:,0],p0=[np.max(sobs_cal[:,0]),const.line_ref,0.2/2.3548,0.0],maxfev=5000)
        spec_1pix[0]=gfit[1]                                                           ## line core wavelength
        spec_1pix[8]=2.3548*gfit[2]                                                    ## Total line width [nm]; 2*np.sqrt(2*np.log(2))==2.3548
    elif gaussfit == 0:                                                                ## CDF only version
        spec_1pix[0]=tmp[4]                                                            ## line core wavelength
        spec_1pix[8]=tmp[5]-tmp[3]                                                     ## Total line width [nm]
    
    print("{:4.6f}".format(time.time()-start0),' gf')

    ## Doppler shifts
    spec_1pix[1]=spec_1pix[0]-const.line_ref                                           ## line  shift from reference position [nm]
    spec_1pix[2]=spec_1pix[1]*const.l_speed*1e-3/const.line_ref                        ## shift converted to velocities; km*s^-1
    
    ## record the intensity of the central wavelength for IQU profiles. 
    #spec_1pix[3]=(sobs_cal[tmp7,0]+sobs_cal[tmp7+1,0])/2.                              ## Stokes I core intensity
    #spec_1pix[4]=(sobs_cal[tmp7,1]+sobs_cal[tmp7+1,1])/2.                              ## Stokes Q core intensity
    #spec_1pix[5]=(sobs_cal[tmp7,2]+sobs_cal[tmp7+1,2])/2.                              ## Stokes U core intensity
    spec_1pix[3:6]=(sobs_cal[tmp7,0:3]+sobs_cal[tmp7+1,0:3])/2.                         ## Stokes IQU core intensity
    #spec_1pix[3:6]=np.mean(sobs_cal[tmp7:tmp7+1,0:3])                                   ## Stokes IQU core intensity
    print("{:4.6f}".format(time.time()-start0),' iqu')
    ## Stokes V Intensity; It will count the min/max counts of the first (left) lobe and will not match wavelength position of the other 3 quantities.
    a=np.argwhere(sobs_cal[:,3] == np.min(sobs_cal[:,3]))[0][0]                           ## check where the negative V lobe is located with respect to the line core position.
    if a <= tmp7:
        spec_1pix[6]=(sobs_cal[a,3])                                                   ## is the negative V lobe is to the left? then Stokes V has a -+ shape
    else:
        b=np.argwhere(sobs_cal[:,3] == np.max(sobs_cal[:,3]))[0][0]
        spec_1pix[6]=(sobs_cal[b,3])                                                   ## Otherwise, Stokes V has a +- shape; 
    print("{:4.6f}".format(time.time()-start0),' v')
    ##background counts
    spec_1pix[7]=background[0]                                                         ## background intensity of stokes I. background in all other Stokes components is similar;
    
    ##line widths (non-thermal)
    spec_1pix[9]=np.sqrt( (((((spec_1pix[8]*1e-9)**2)*(const.l_speed**2))\
        -(((instwidth*1e-9)**2)*((const.line_ref*1e-9)**2)))/(4*np.log(2)*((const.line_ref*1e-9)**2)))\
        -(const.kb*(10.**const.ion_temp)/const.ion_mass))*const.line_ref/const.l_speed ## Non-thermal component of the line width; dependent on instwidth
    
    ## compute the polarization quantities
    spec_1pix[10] =np.sqrt(sobs_tot[1]**2+sobs_tot[2]**2)/sobs_tot[0]                  ## fraction of linear polarization with respect to intensity
    spec_1pix[11] =np.sqrt(sobs_tot[1]**2+sobs_tot[2]**2\
        +sobs_tot[3]**2)/sobs_tot[0]                                                   ## fraction of total polarization with respect to intensity
    print("{:4.6f}".format(time.time()-start0),' fin')
    return spec_1pix
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_matchiquv(sobs_1pix,yobs_1pix,aobs_1pix,database_sel,dbhdr,rms,nsearch,maxchisq,bcalc,reduced,verbose):
## main solver for the geometry and magnetic field strength version for full stokes vector
## Returns matched database index, double line IQUV vector, and chi^2 fitting residual
## Returns matched observation physics:
## Y(radial) position and X(LOS) position;
## B_phi and B_theta angles in L.V.S geometry;
## Bx, By, Bx in LOS geometry (L.V.S. transform to cartesian);
## B field strength.

    ## unpack dbcgrid parameters from the database accompanying db.hdr file 
    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
        bthetamin, bthetamax, nline, wavel  = dbhdr
    ####NO OBSERVATION --> NO RUN! ###########    
    ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
    ##returns an array-like 0 vector of the dimensions of the requested output.
    ## the index is set to -1 to warn about the missing entry
    if np.isnan(sobs_1pix).all() or np.count_nonzero(sobs_1pix) == 0:
        nullout = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 2: print("WARNING (CLEDB_MATCH): No observation in voxel!")
        return nullout,np.zeros((nsearch,8),dtype=np.float32)

    ##  Read the database in full or reduced mode  
    if reduced == True:
        ## database_sel becomes the reduced database in this case
        ## outredindex is the corresponding index for reduced, this is different from the full index. they need to be matched later.
        ## cledb_getsubset will reshape database_sel to [index,nline*4] shape
        database_sel,outredindex = cledb_getsubset(sobs_1pix,dbhdr,database_sel,nsearch,verbose) 
        if verbose >= 3:print("CLEDB_MATCH: using reduced DB:",database_sel.shape,outredindex.shape)
        
    else:
        outredindex = np.empty((nbphi,nsearch),dtype=np.float32)                       ## In this branch, outredindex is never used, but numba does not properly compile because the array is presumably presumably used for matching.
        database_sel = np.reshape(database_sel,(ned*ngx*nbphi*nbtheta,8))              ## Reshape is found to be the fastest python/numpy/numba method to reorganize array dimensions.
        outargs=np.argwhere(np.sign(database_sel[:,3]) == np.sign(sobs_1pix[3]) )[:,0]     ## keeps the argument for the reduced entry based on the sign of Stokes V.
        database_sel = database_sel[np.sign(database_sel[:,3]) == np.sign(sobs_1pix[3])]   ## check and presort the sign along the first line 

        if verbose >= 2:
            if (database_sel.shape[0] > 1000000):
                print("WARNING (CLEDB_MATCH): Full database match will be significantly slower with no proven benefits!")
            if verbose >= 3:print("CLEDB_MATCH: using full DB:",database_sel.shape,outredindex.shape)

    ##normalize the input data to the strongest component
    norm_fact=sobs_1pix[np.argwhere(sobs_1pix == np.max(sobs_1pix))[0][0]]
    sobs_1pix[:]=sobs_1pix[:]/norm_fact
        
    ## Geometric solution is based on a reduced chi^2 measure fit.
    ## match sobs data with database_sel using the reduced chi^2 method
    ## If Observation has Stokes V, we include it in the difference for the chi^2 measurement
    diff = np.zeros((database_sel.shape[0],8),dtype=np.float32)
    diff = (database_sel[:,:8] - sobs_1pix) /rms#/ (rms / norm_fact)
    ndata = 4*2-1         ## Number of observables

          
    denom = ndata-4           ## Denominator used below in reduced chi^2      ## 4 = number of degree of freedom in model: ne, x, bphi, btheta
    diff *= diff              ## pure python multiplication was found to be faster than numpy or any power function applied.
    chisq = np.zeros((diff.shape[0]),dtype=np.float32)
    np.round_( np.sum( diff, axis=1)/denom,15,chisq)
    ## Unorthodox definition: chisq needs to be initialized separately, because np.round_ is not supported as a addressable function
    ## This is the only numba compatible implementation possible with current numba implementation.
    ## Precision is up to 15 decimals; Significant truncation errors appear if this is not enforced. 
    ## if no rms and counts are present, the truncation is beyond a float32 without this decimal fix.

    ## Need to use a manual numba compatible fast sorting as np.argpartition is not numba implemented!
    ## cledb_partsort is a simple parallel function that produces an output similar to np.argpartition with sort.
    ## it just does a standard sorting search, but sorts just the first nsearch elements, making it much faster than np.argsort for few solutions.
    ## for a lot of solutions, a full sort will perform better
    if nsearch <= 50:
        asrt = cledb_partsort(chisq,nsearch) 
    else:
        if verbose >= 2: print("WARNING (CLEDB_MATCH): High number of solutions requested. Expect slow runtime. Using a full array sort!")
        asrt =np.argsort(chisq)[0:nsearch]
    
    ## the for loop will compute the first nsearch solutions;
    ## OR
    ## solutions that have chi^2 less than maxchisq; 
    ## whichever comes first!!

    ## initialize return indices and chisq of nearest solutions
    si = 0
    ix = asrt[0]
    ixr = asrt[0]
    ixrchisq = chisq[ixr]

    ## returns an array-like 0 vector of the dimensions of the requested output.
    ## the index is set to -1 to warn about the missing entry
    ## the index is set to -1 to warn about the missing entry
    if chisq[ixr] > maxchisq:
        nullout = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 2: print("WARNING (CLEDB_MATCH): No solutions compatible with the maximum Chi^2 constraint.")
        return nullout,np.zeros((nsearch,8),dtype=np.float32) 

    ## arrays for storing the results
    out=np.zeros((nsearch,11),dtype=np.float32)
    out[:,0]=np.full(nsearch,-1)                  ## start all indexes as not found, then update as you go
    smatch=np.zeros((nsearch,8),dtype=np.float32)

    ## calculate the physics from the database and fill the output arrays
    for si in range(nsearch):

        ixr = asrt[si]                    ## index in (presumably reduced) databasex
        ixrchisq = chisq[ixr]             ## the chisq should correspond to the data array(ixr), be it reduced or not
        if ixrchisq <= maxchisq:          ## Are we still computing this si entry?

            # Magnetic field strength from ratio of database with V/I ratios or wave data of observations and database
            if bcalc == 0: ## using first line
                bfield = sobs_1pix[3]/(database_sel[ixr,3]+1e-8) ## for division operation precision when database bfields are close to 0;
            if bcalc == 1: ## using second line
                bfield = sobs_1pix[7]/(database_sel[ixr,7]+1e-8)
            if bcalc == 2: ## using the average of the two lines
                bfield = 0.5*(sobs_1pix[3]/(database_sel[ixr,3]+1e-8) + sobs_1pix[7]/(database_sel[ixr,7]+1e-8) )
            ## No bcalc == 3 in a full stokes inversion 

            ## matching profiles to compare with the original observation
            smatch[si,:] = database_sel[ixr,:]                                              ## if database is reduced ixr is the right index; otherwise ixr=ix
            smatch[si,:] = cledb_quderotate(smatch[si,:],aobs_1pix,norm_fact)

            # here if reduced we must get the original value of ix  to write to output (if required!).
            if reduced == True:
                i,j,k,l = cledb_params(ixr,np.array((dbcgrid[0],dbcgrid[1],dbcgrid[2],outredindex.shape[1],0,0,0,0,0,0,0,0,0,0),dtype=np.float32))   ## extremely ugly but fast. 0 are just to reproduce the shape of dbcgrid
                ix = cledb_invparams(i,j,k,np.int64(outredindex[k,l]),dbcgrid)                                                      ## = original value ix using the reduced outredindex to replace "l"
                ##update the inversion output array
                out[si,0]=ix
                out[si,1]=ixrchisq
                out[si,2:]=cledb_phys(ix,yobs_1pix,aobs_1pix,dbhdr,bfield)
            else:
                ix = asrt[si]                     ## this is updated if reduced is used
                ##update the inversion output array
                out[si,0]=outargs[ix]                               ### uses args to produce the correct physics
                out[si,1]=ixrchisq
                out[si,2:]=cledb_phys(outargs[ix],yobs_1pix,aobs_1pix,dbhdr,bfield)

    ######################################################################
    ## [placeholder for issuemask]

    return out,smatch
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_matchiqud(sobs_1pix,sobsd_1pix,yobs_1pix,aobs_1pix,database_sel,dbhdr,rms,nsearch,maxchisq,bcalc,reduced,verbose):
## main solver for the geometry and magnetic field strength -- Version with linear polarization+doppler data.
## Returns matched database index, double line IQU vector, and chi^2 fitting residual
## Returns matched observation physics:
## Y(radial) position and X(LOS) position;
## B_phi and B_theta angles in L.V.S geometry;
## Bx, By, Bx in LOS geometry (L.V.S. transform to cartesian);
## B field strength.

    ## unpack dbcgrid parameters from the database accompanying db.hdr file 
    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
        bthetamin, bthetamax, nline, wavel  = dbhdr
    ####NO OBSERVATION --> NO RUN! ###########    
    ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
    ##returns an array-like 0 vector of the dimensions of the requested output.
    ## the index is set to -1 to warn about the missing entry
    if np.isnan(sobs_1pix).all() or np.count_nonzero(sobs_1pix) == 0:
        nullout = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 2: print("WARNING (CLEDB_MATCH): No observation in voxel!")
        return nullout,np.zeros((nsearch,8),dtype=np.float32)

    ##  Read the database in full or reduced mode  
    if reduced == True:
        ## database_sel becomes the reduced database in this case
        ## outredindex is the corresponding index for reduced, this is different from the full index. they need to be matched later.
        ## cledb_getsubset will reshape database_sel to [index,nline*4] shape
        database_sel,outredindex = cledb_getsubset(sobs_1pix,dbhdr,database_sel,nsearch,verbose) 
        if verbose >= 3:print("CLEDB_MATCH: using reduced DB:",database_sel.shape,outredindex.shape)
        
    else:
        outredindex = np.empty((nbphi,nsearch),dtype=np.float32)                ## In this branch, outredindex is never used, but numba does not properly compile because the array is presumably presumably used for matching.
        database_sel = np.reshape(database_sel,(ned*ngx*nbphi*nbtheta,8))       ## Reshape is found to be the fastest python/numpy/numba method to reorganize array dimensions.
        ## Different from IQUV implementation
        outargs=np.argwhere(np.sign(database_sel[:,3]) == np.sign(sobsd_1pix[2]+1e-7) )[:,0]   ## keeps the argument for the reduced entry based on the sign from doppler waves.
        database_sel = database_sel[np.sign(database_sel[:,3]) == np.sign(sobsd_1pix[2]+1e-8)]  ## check the sign of the stokes observation used to compute blos


        if verbose >= 2:
            if (database_sel.shape[0] > 1000000):
                print("WARNING (CLEDB_MATCH): Full database match will be significantly slower with no proven benefits!")
            if verbose >= 3:print("CLEDB_MATCH: using full DB:",database_sel.shape,outredindex.shape)

    ##normalize the input data to the strongest component
    norm_fact=sobs_1pix[np.argwhere(sobs_1pix == np.max(sobs_1pix))[0][0]]
    sobs_1pix[:]=sobs_1pix[:]/norm_fact
        
    ## Geometric solution is based on a reduced chi^2 measure fit.
    ## match sobs data with database_sel using the reduced chi^2 method                                                
    ## Different from IQUV implementation
    ## We do not use Stokes V to define the geometry. Therefore we use a subset (1,2,4,5,6) 
    ## of Stokes parameters QU (line 1) and IQU (line 2); total=5
    diff = np.zeros((database_sel.shape[0],6),dtype=np.float32)
    ## These two below lines are the slowest of this function due to requiring two operations.
    ## A better numba COMPATIBLE alternative was not found
    diff[:,0:2] = (database_sel[:,1:3] - sobs_1pix[1:3]) /rms[1:3]#/ np.abs(rms[1:3])
    diff[:,2:5] = (database_sel[:,4:7] - sobs_1pix[4:7]) /rms[4:7]#/ np.abs(rms[4:7])
    ndata = 4*2-1-2          ## Number of observables for geometry solver ## ndata = -2 comes from not using the two V components (missing in IQUD).    

    denom = ndata-4           ## Denominator used below in reduced chi^2      ## 4 = number of degree of freedom in model: ne, x, bphi, btheta
    diff *= diff              ## pure python multiplication was found to be faster than numpy or any power function applied.
    chisq = np.zeros((diff.shape[0]),dtype=np.float32)
    np.round_( np.sum( diff, axis=1)/denom,15,chisq)
    ## Unorthodox definition: chisq needs to be initialized separately, because np.round_ is not supported as a addressable function
    ## This is the only numba compatible implementation possible with current numba implementation.
    ## Precision is up to 15 decimals; Significant truncation errors appear if this is not enforced. 
    ## if no rms and counts are present, the truncation is beyond a float32 without this decimal fix.

    ## Need to use a manual numba compatible fast sorting as np.argpartition is not numba implemented!
    ## cledb_partsort is a simple parallel function that produces an output similar to np.argpartition with sort.
    ## it just does a standard sorting search, but sorts just the first nsearch elements, making it much faster than np.argsort for few solutions.
    ## for a lot of solutions, a full sort will perform better
    if nsearch <= 50:
        asrt = cledb_partsort(chisq,nsearch) 
    else:
        if verbose >= 2: print("WARNING (CLEDB_MATCH): High number of solutions requested. Expect slow runtime. Using a full array sort!")
        asrt =np.argsort(chisq)[0:nsearch]
    
    ## the for loop will compute the first nsearch solutions;
    ## OR
    ## solutions that have chi^2 less than maxchisq; 
    ## whichever comes first!!

    ## initialize return indices and chisq of nearest solutions
    si = 0
    ix = asrt[0]
    ixr = asrt[0]
    ixrchisq = chisq[ixr]

    ## returns an array-like 0 vector of the dimensions of the requested output.
    ## the index is set to -1 to warn about the missing entry
    if chisq[ixr] > maxchisq:
        nullout = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 2: print("WARNING (CLEDB_MATCH): No solutions compatible with the maximum Chi^2 constraint.")
        return nullout,np.zeros((nsearch,8),dtype=np.float32) 

    ## arrays for storing the results
    out=np.zeros((nsearch,11),dtype=np.float32)
    out[:,0]=np.full(nsearch,-1)                  ## start all indexes as not found, then update as you go
    smatch=np.zeros((nsearch,8),dtype=np.float32)

    ## calculate the physics from the database and fill the output arrays
    for si in range(nsearch):

        ixr = asrt[si]                    ## index in (presumably reduced) databasex
        ixrchisq = chisq[ixr]             ## the chisq should correspond to the data array(ixr), be it reduced or not
        if ixrchisq <= maxchisq:          ## Are we still computing this si entry?

            # Magnetic field strength from ratio of database with sign from wave tracking
            if bcalc == 3: ## using the fieldstrength from the wave tracking
                bfield = sobsd_1pix[0]*np.sign(sobsd_1pix[2]+1e-8)

            ## matching profiles to compare with the original observation
            smatch[si,:] = database_sel[ixr,:]                                              ## if database is reduced ixr is the right index; otherwise ixr=ix
            smatch[si,:] = cledb_quderotate(smatch[si,:],aobs_1pix,norm_fact)

            # here if reduced we must get the original value of ix  to write to output (if required!).
            if reduced == True:
                i,j,k,l = cledb_params(ixr,np.array((dbcgrid[0],dbcgrid[1],dbcgrid[2],outredindex.shape[1],0,0,0,0,0,0,0,0,0,0),dtype=np.float32))   ## extremely ugly but fast. 0 are just to reproduce the shape of dbcgrid
                ix = cledb_invparams(i,j,k,np.int64(outredindex[k,l]),dbcgrid)                                                      ## = original value ix using the reduced outredindex to replace "l"
                ##update the inversion output array
                out[si,0]=ix
                out[si,1]=ixrchisq
                out[si,2:]=cledb_phys(ix,yobs_1pix,aobs_1pix,dbhdr,bfield)
            else:
                ix = asrt[si]                     ## this is updated if reduced is used
                ##update the inversion output array
                out[si,0]=outargs[ix]                               ### uses outargs to produce the correct physics
                out[si,1]=ixrchisq
                out[si,2:]=cledb_phys(outargs[ix],yobs_1pix,aobs_1pix,dbhdr,bfield)

    ######################################################################
    ## [placeholder for issuemask]

    return out,smatch
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_getsubset(sobs_1pix,dbhdr,database_sel,nsearch,verbose): 
## returns a subset of the sdb array compatible with the height in yobs

    ## unpack dbcgrid parameters from the database accompanying db.hdr file
    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
    bthetamin, bthetamax, nline, wavel  = dbhdr 

    ## Create the reduced arrays for analysis; we don't need more than the desired nsearch subsets
    datasel = np.zeros((ned,ngx,nbphi,2*nsearch,8),dtype=np.float32)
    outredindex = np.zeros((nbphi,2*nsearch),dtype=np.float32)
    ##NOTE: the sorting done here only takes into account the linear polarization tangent of the observation. This sorting is not 1:1 equivalent with the main chi^2 sorting.
    ##      To be sure all compatible solutions are captured nsearch*4 solutions must be enforced here, and further refined in cledb_match

    ## database: indexes of separate the CLE array calculations for reduction
    kk=np.arange(0,nbphi)
    bphir  = bphimin + kk*(bphimax-bphimin)/(nbphi-1)         ## cle phi array

    ll=np.arange(0,nbtheta)
    bthetar=bthetamin + ll*(bthetamax-bthetamin)/(nbtheta-1)  ## cle theta array
    tt = np.tan(bthetar)                                      ## cle tan theta array

    ## Observation: compute the PHI_B and its degenerate tangents
    phib_obs = - 0.5*np.arctan2(sobs_1pix[2],sobs_1pix[1])    ## tan Phi_B = sin phi * tan theta
    tphib_obs = np.tan(phib_obs)                              ## here is tan Phi_B:
    tphib_deg_obs = np.tan(phib_obs+np.pi/2.)                 ## and its degenerate branch

    # Find those indices compatible with phib observed
    for ir in range(bphir.shape[0]):                          ## loop over bphi in cle frame
        ttp = tt * np.sin(bphir[ir])
        diffa = np.abs(tphib_obs - ttp)                       ## this is an array over btheta at each bphi
        diffb = np.abs(tphib_deg_obs - ttp)                   ## this is an array over btheta at each bphi (degenerate branch)
        srta=cledb_partsort(diffa,nsearch)                    ## NOTE: no speed gain to use np.sort here as the arrays are small.
        srtb=cledb_partsort(diffb,nsearch)
        ## advanced slicing is not available, the for jj enumeration comes from numba requirements
        if ir + srta[0] > 0 or  ir + srtb[0] > 0:                                  ## important to avoid phi=0 AND theta = 0 case
            for jj in range(nsearch):                                              ## NOTE: nsearch = srt.shape[0]
                datasel[:,:,ir,jj,:]           = database_sel[:,:,ir,srta[jj],:]   ## Record those indices compatible with phib observed
                outredindex[ir,jj]             = srta[jj]
                datasel[:,:,ir,jj+(nsearch),:] = database_sel[:,:,ir,srtb[jj],:]   ## Record those indices compatible with phib observed (degenerate branch)
                outredindex[ir,jj+(nsearch)]   = srtb[jj]

    ##Note: the above block is both faster and more accurate than concatenating the diff arrays + sorting only once + one subscription for datasel and outredindex.

    # now work with reduced dataset with nbtheta replaced by nsearch
    # the full dimensions of the database are no longer needed. It is converted to a [index,8] shape
    if verbose >= 3: 
        print('Search over theta reduced by a factor: ', np.int32(nbtheta/nsearch),". New db size: (",ned*ngx*nbphi*nsearch,",8)")
        #if verbose >= 3: print(dt,' SECONDS FOR REDUCE (loop phi) CALC') 
    return np.reshape(datasel,(ned*ngx*nbphi*2*nsearch,8)),outredindex
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_quderotate(dbarr,ang,nf):
## derotates the matched database entry to make it compatible with the observation; 
## same Mueller transform as obs_qurotate, but applied with the oposite angle as we just want to compare the DB stokes profiles with the observed ones.

    arr=np.copy(dbarr)*nf ## to not alter exterior arrays; use dbarr for updating as to not use modified arr[1] and arr[5] values for arr[2] and arr[6]
    arr[1]=dbarr[1]*nf*np.cos(-2.*ang) - dbarr[2]*nf*np.sin(-2.*ang)
    arr[2]=dbarr[1]*nf*np.sin(-2.*ang) + dbarr[2]*nf*np.cos(-2.*ang)
    arr[5]=dbarr[5]*nf*np.cos(-2.*ang) - dbarr[6]*nf*np.sin(-2.*ang)
    arr[6]=dbarr[5]*nf*np.sin(-2.*ang) + dbarr[6]*nf*np.cos(-2.*ang)

    return arr
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_partsort(arr,nsearch):
## cledb_partsort is a parallel function that produces an output similar to np.argpartition with sort.
## Simplest possible substitution partial sort. It just returns the first nsearch sorted indexes similar to np.argpartition.
## It is numba non-python compatible!

    asrt=np.zeros((nsearch),dtype=np.int64)
    arr_temp=np.copy(arr) ## don't change the input array inside; numba will complain of changing an upstream array that you don't return back.
    for i in range (nsearch):
        a=np.argmin(arr_temp[i:])
        b=np.argwhere(arr == arr_temp[i:][a])[0]
        d=0
        for j in range(b.shape[0]):
            if b[j] in asrt[:i]:
                d+=1
        asrt[i]=b[d]
        sort_temp=arr_temp[i]
        arr_temp[i]=arr_temp[i:][a]
        arr_temp[i:][a]=sort_temp

    return asrt         ##works like a charm!
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_params(index,dbcgrid):
## This function is used with the database header to help compute the physics associated to the i, j, k, l entry
## for index, get i,j,k,l indices in a database

    ned=np.int32(dbcgrid[0]) 
    ngx=np.int32(dbcgrid[1])
    nbphi=np.int32(dbcgrid[2])
    nbtheta=np.int32(dbcgrid[3])
    n5=( ngx*nbphi*nbtheta)
    n4=(     nbphi*nbtheta)
    n3=(           nbtheta)

    i = index   //        n5
    j = index   -     i * n5
    j = j      //         n4
    k = index   -     i * n5 - j * n4
    k = k      //         n3
    l=  index   -     i * n5 - j * n4 - k * n3

    return i,j,k,l
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_invparams(i,j,k,l,dbcgrid):
## This function is used with the database header to help compute the physics associated to the index entry
## Reverse function of params_par## 
## for i,j,k,l in database, get index

    ned=np.int32(dbcgrid[0]) 
    ngx=np.int32(dbcgrid[1])
    nbphi=np.int32(dbcgrid[2])
    nbtheta=np.int32(dbcgrid[3])

    return np.int32(i*ngx*nbphi*nbtheta + j*nbphi*nbtheta + k*nbtheta + l)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_elecdens(r):
## computes an electron radial density estimation using the Baumbach formulation

    baumbach = 1.e8*(0.036/r**1.5 + 1.55/r**6.)
    hscale=   7.18401074e-02  # 50 Mm

    return np.float32(3.e8*np.exp(- (r-1.)/hscale) + baumbach)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down 
def cledb_phys(index,gy,ga,dbhdr,b):
## Returns the lvs and LOS geometry and magnetic field physics
## this is kept separate from cledb_physlvs because it computes projections transformations of the database variables recovered via cledb_physlvs

    phs=cledb_physlvs(index,gy,dbhdr)
    bphi=phs[3]#np.abs(phs[3] - ga)                            ## derotate by azimuth rotation(ga) applied to Q and U profiles; does it not need disentangling?
    btheta=phs[4]
    bx=np.abs(b)*np.sin(btheta)*np.cos(bphi)
    by=np.abs(b)*np.sin(btheta)*np.sin(bphi)
    bz=np.abs(b)*np.cos(btheta)

    return np.array((phs[0],phs[1],phs[2],b,bphi,btheta,bx,by,bz),dtype=np.float32)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down 
def cledb_physlvs(index,gy,dbhdr):
## Helper for phys_par; returns the LVS physics and compatible observation geometry.
## this is the primary function that retrieves physics parameters from the database index.

    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
    bthetamin, bthetamax, nline, wavel  = dbhdr

    i,j,k,l = cledb_params(index,dbcgrid)

    gx = gxmin + j*(gxmax-gxmin)/(ngx-1)
    bphi  = bphimin + k*(bphimax-bphimin)/(nbphi-1)
    btheta  = bthetamin + l*(bthetamax-bthetamin)/(nbtheta-1)
    ed = np.float32(xed[i]* cledb_elecdens(np.sqrt(gy*gy+gx*gx)))    # log of Ne

    return np.array((np.log10(ed),gy,gx,bphi,btheta),dtype=np.float32)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_cdf(spectr):
## computes the cdf distribution for a spectra

    cdf=np.zeros((spectr.shape[0]),dtype=np.float32)
    for i in prange (0,spectr.shape[0]):
        cdf[i]=np.sum(spectr[0:i+1])        ## need to check if should start at 0 or not
    cdf=cdf[:]/cdf[-1]                      ## norm the cdf to simplify interpretation
    return cdf
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down 
# ## Simple Gaussian Function for spectra fitting
def obs_gaussfit(x,ymax,mean,sigma, offset):
# ## Simple Gaussian Function
    return ymax*np.exp(-(x-mean)**2/(2*sigma**2)) + offset
###########################################################################
###########################################################################
