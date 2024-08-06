# -*- coding: utf-8 -*-

###############################################################################
### Parallel implementation of utilities for CLEDB_PROC                     ###
###############################################################################

#Contact: Alin Paraschiv, National Solar Observatory 
#         arparaschiv@nso.edu
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

## Functions in the time module are not understood/supported by numba. enabling them forces the compiler to revert to object mode and throw warnings. 
## Full Parallelization will generally not be possible while time functions are enabled.



#########################################################################
### Needed imports ######################################################
#########################################################################

#
# Needed libraries
#
import numpy as np
from   numba import jit,njit,prange
import scipy.stats as sps
from   scipy.optimize import curve_fit
import time
import constants
import ctrlparams 
params=ctrlparams.ctrlparams()    ## just a shorter label

## optional libraries
# import glob
# import os
# import sys
# import numexpr as ne

###########################################################################
###########################################################################
### Main Modules ##########################################################
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,cache=params.jitcache)
def cledb_invproc(sobs_totrot,sobs_dopp,database,db_enc,yobs,aobs,rms,dbhdr,keyvals,nsearch,maxchisq,bcalc,iqud,reduced,verbose):

    if verbose >= 1: 
        print('--------------------------------------\n----CLEDB_INVPROC - INVERSION START---\n--------------------------------------')
        #if verbose >= 2: start=time.time()  ##Numpy non-python incompatible with time module.

    ######################################################################
    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny  = keyvals[0:2]
    
    ##Output variables
    invout = np.zeros((nx,ny,nsearch,11),dtype=np.float32)
    sfound = np.zeros((nx,ny,nsearch,8),dtype=np.float32)
    
    ######################################################################
    ## some checks for potential problems with cledb_invproc input data and control parameter settings
    
    ## Number of degrees of freedom >4; we require a two line full IQU+ V or D observation.
    ## assume the last dimension of sobs is the number of observables; two lines = 8 for IQUV or 6 obs +2 columns that are 0 for IUQD
    ## Database needs to match the observation
    if sobs_totrot.shape[-1] != 8 or database[0].shape[-1] != 8: 
        if verbose >=1:
            print('CLEDB_INVPROC: FATAL! Must have a two-line observation and a corresponding two-line database; Aborting!')
            print("Shapes are: ", sobs_totrot.shape," and ",database[0].shape)
        invout[:,:,:,0]=np.full((nx,ny,nsearch),-1) ### index is -1 for failed runs!
        return invout,sfound
    
    ### NO wave observation and matching for IQUD --> NO RUN  ######
    if (iqud == True and bcalc != 3) or (iqud != True and bcalc == 3):
        invout[:,:,:,0]=np.full((nx,ny,nsearch),-1) ### index is -1 for failed runs!
        if verbose >= 1: 
            print("CLEDB_INVPROC: FATAL! Field strength calculation incompatible with requested input data! Check bcalc and iqud keywords; Aborting!")
        return invout,sfound
    
    if reduced == True:
        if verbose >= 2: 
            print("CLEDB_INVPROC: WARNING! Using a reduced database!")
            print('Search over theta reduced by a factor: ', np.int32(dbhdr[4]/nsearch),". New db size: (",\
                  dbhdr[1]*dbhdr[2]*dbhdr[3]*dbhdr[4],",8) --> (",dbhdr[1]*dbhdr[2]*dbhdr[3]*nsearch,",8)") ##not worth unpacking dbhdr.
    else:
        if verbose >= 2:
            print("CLEDB_INVPROC: using a full database:",len(database[0])/8)
            if (len(database[0]) > 1000000): print("CLEDB_INVPROC: WARNING! Full database match with a large databases will be significantly slower")
    
    if nsearch > 16:    
        if verbose >= 2: 
            print("CLEDB_INVPROC: WARNING! High number of solutions requested. Expect slower runtime proportional to nsearch control parameter.")        
            if nsearch > 100 : 
                print("CLEDB_INVPROC: A VERY HIGH number of solutions requested. Using a full array sort! This will take some time to finish.")     
    ######################################################################
    
    ######################################################################
    ## Main loops to do the matching pixelwise
    ## We use the iqud keyword to run the fucntion that matches either IQUV or IQUD data.
    if iqud == True:
        if verbose >= 3:         ## Heavy print output
            for xx in range(nx): 
                print("CLEDB_INVPROC (IQUD): Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                for yy in prange(ny): 
                    print("                 Executing calculation: ",yy," of ",ny)
                    invout[xx,yy,:,:],sfound[xx,yy,:,:] = cledb_matchiqud(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],\
                                                                          database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],nsearch,maxchisq,bcalc,reduced,verbose)        
        elif verbose >= 1:       ## Some print output   
            for xx in range(nx): #range(301,303): #
                print("CLEDB_INVPROC (IQUD): Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                for yy in prange(ny): #prange(100,200): #
                    invout[xx,yy,:,:],sfound[xx,yy,:,:] = cledb_matchiqud(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],\
                                                                          database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],nsearch,maxchisq,bcalc,reduced,verbose)
        else:                    ## No print output
            for xx in range(nx):
                for yy in prange(ny):
                    invout[xx,yy,:,:],sfound[xx,yy,:,:] = cledb_matchiqud(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],\
                                                                          database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],nsearch,maxchisq,bcalc,reduced,verbose)
    else:
        if verbose >= 3:         ## Heavy print output  
            for xx in range(nx):
                print("CLEDB_INVPROC (IQUV): Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                for yy in prange(ny):
                    print("                 Executing calculation: ",yy," of ",ny)
                    invout[xx,yy,:,:],sfound[xx,yy,:,:] = cledb_matchiquv(sobs_totrot[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],\
                                                                          dbhdr,rms[xx,yy,:],nsearch,maxchisq,bcalc,reduced,verbose)        
        elif verbose >= 1:       ## Some print output       
            for xx in range(nx): #range(391,392): #
                print("CLEDB_INVPROC (IQUV): Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                for yy in prange(ny): #prange(100,200): #
                    # if xx == 301 and yy == 105:
                    #     sobs_totrot[xx,yy,:]=np.array((1.00000000e+00, 3.16772669e-04, 1.52991570e-03, 5.63795435e-06, 6.24271335e-01, 1.66090059e-05, 8.02164495e-05, 3.53448432e-06),dtype=np.float32) ##5,25,52,70 or 5269750
                    #     print(database[db_enc[xx,yy]][5,25,52,70,:])
                    #     print(sfound[xx,yy,:,:]/sfound[xx,yy,:,0])
                    #     print(invout[xx,yy,:,0:2])
                    # else:
                    #     invout[xx,yy,:,:],sfound[xx,yy,:,:]=cledb_matchiquv(sobs_totrot[xx,yy,:],sobs_dopp[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],dbhdr,rms[xx,yy,:],\
                    #                                                     nsearch,maxchisq,bcalc,iqud,reduced,verbose)             
                    invout[xx,yy,:,:],sfound[xx,yy,:,:] = cledb_matchiquv(sobs_totrot[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],\
                                                                          dbhdr,rms[xx,yy,:],nsearch,maxchisq,bcalc,reduced,verbose)
        else:                    ## No print output
            for xx in range(nx):
                for yy in prange(ny):
                    invout[xx,yy,:,:],sfound[xx,yy,:,:] = cledb_matchiquv(sobs_totrot[xx,yy,:],yobs[xx,yy],aobs[xx,yy],database[db_enc[xx,yy]],\
                                                                          dbhdr,rms[xx,yy,:],nsearch,maxchisq,bcalc,reduced,verbose)
                    
    ######################################################################
    ## [placeholder for issuemask]                    
    
    if verbose >=1:
        #if verbose >= 2: print("{:4.6f}".format(time.time()-start),' SECONDS FOR DB INVERSION PROCESSING')
        print('--------------------------------------\n--CLEDB_INVPROC - INVERSION FINALIZED-\n--------------------------------------')

    return invout,sfound
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache) ##Object mode because of time module. not faster in non-python!
def blos_proc(sobs_tot,rms,keyvals,consts,params):
## Uses the "improved" magnetograph formulation in eq 40 Casini &Judge 99, eq 14 of Plowman 2014, and eq 17 and 18 of Dima & Schad 2020
## Follows the discussion in the three papers and adopts different analythical implementations based on the line combination used
## As shown in the papers the magnetograph formulation is not precise in terms of recovering the LOS magnetic field.
## differences of the order of $\pm$ 2 times actual values  based on the atomic alignment, F factor, and LOS angle $\theta$ can manifest.

## script will produce a set of 2 times degenerate magnetograph along with a classic magnetograph and a field azimuth for each fed line/observation.

    if params.iqud == True:              ## No Los analytical expression without Stokes V
        print("FATAL! No Stokes V data available to compute analytical line-of-sight projected magnetic fields.")
        return 0

    else:
        if params.verbose >= 1: 
            print('--------------------------------------\n---BLOS_PROC: B LOS ESTIMATION START--\n--------------------------------------')
            if params.verbose >= 2: start = time.time()  

        ## unpack needed values from keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
        nx,ny = keyvals[0:2]
        tline = keyvals[4] 

        ## needed array initialization
        blosout = np.zeros((nx,ny,4,len(tline)),dtype=np.float32)

        for zz in range(0,len(tline)):
            ## initialize constants
            const = consts.Constants(tline[zz])                       ## for each line load the correct constants

            ## the list of the F factors from Dima & Schad 2020 for lines of interest are saved in the constants class; 
            ## both Fe XIII and SI IX have F == 0.
            if params.verbose >= 2 and tline[zz] == "si-x_1430":
                print('BLOS_PROC: Constrained magnetograph solutions will have an additional correction applied.')

            ## two degenerate solutions are produced for blos; Indexes 0 and 1
            ## the accurate solution matches sign with \sigma ^2_0 atomic alignment.
            ## the "standard" magnetograph formulation (index 2) represents, in practice, the average accuracy between the two degenerate solutions.
            ## Note: this solution is inaccurate in almost all cases; with the exception of fields tangential to the radial direction.

            ## blos_proc runs fast and does not have subfunctions and their associated errors. There is no need to print each loop iteration
            for xx in range(nx):
                for yy in prange(ny):          
                    lpol                = np.sqrt(sobs_tot[xx,yy,1+(4*zz)]**2 + sobs_tot[xx,yy,2+(4*zz)]**2)         ## linear polarization total
                    blosout[xx,yy,3,zz] = + 0.5 * np.arctan2(sobs_tot[xx,yy,2+(4*zz)],sobs_tot[xx,yy,1+(4*zz)])      ## the azimuthal component (phi)
                    if (sobs_tot[xx,yy,0+(4*zz)] != 0) and (np.isnan(sobs_tot[xx,yy,4*zz:4*(zz+1)]).any() == False): ## captures nans or division by 0
                        ## for corrected magnetograph magnetic field implement eq.17 from Dima & Schad 2020
                        ## where we use sin^Theta_B ==1; theta_b=90 cf chap 4.3 Dima & Schad 2020 
                        ## to minimize deviation from "true" solution when computing the last F ferm dependent correction.
                        ## 1e9 converts SI constants to nm to divide by the reference wavelength also in nm
                        blosout[xx,yy,0,zz] = (const.planckconst/const.bohrmagneton*const.l_speed*1.e9/(const.line_ref**2)) *\
                        (-sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*(sobs_tot[xx,yy,0+(4*zz)] - lpol) - 0.66*const.F_factor*(lpol/1.)))  ## deg. solution +
                        blosout[xx,yy,1,zz] = (const.planckconst/const.bohrmagneton*const.l_speed*1.e9/(const.line_ref**2)) *\
                        (-sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*(sobs_tot[xx,yy,0+(4*zz)] + lpol) + 0.66*const.F_factor*(lpol/1.)))  ## deg. solution -
                        blosout[xx,yy,2,zz] = (const.planckconst/const.bohrmagneton*const.l_speed*1.e9/(const.line_ref**2)) *\
                        (-sobs_tot[xx,yy,3+(4*zz)] / (const.g_eff*sobs_tot[xx,yy,0+(4*zz)]))                                           ## magnetograph
                    #else:
                    #    blosout[xx,yy,0:3,zz]==np.nan  ## else branch not needed. blosout is already nan in pixels where field can't be computed

        ######################################################################
        ## [placeholder for issuemask]

        if params.verbose >= 1:
            if params.verbose >= 2: print("{:4.6f}".format(time.time()-start),' SECONDS FOR TOTAL BLOS PROCESSING')
            print('--------------------------------------\n---BLOS_PROC: B LOS ESTIMATION END----\n--------------------------------------')   

        return blosout
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)  ##object mode because of ##CDF_STATISTICS
def spectro_proc(sobs_in,sobs_tot,rms,background,keyvals,consts,params):
## calculates 12 spectroscopic products
## Background is used both as an output product and as a preprocess parameter. More details in cdf_statistics comments.


    if params.integrated == True: ## No spectroscopy on integrated data.
        print("FATAL! Integrated data for IQUD does not have a wavelength dimension. No spectroscopy products to compute!")            
        return 0
    
    else:
        if params.verbose >= 1: 
            print('--------------------------------------\n---SPECTRO_PROC - SPECTROSCOPY START--\n--------------------------------------')
            if params.verbose >= 2: start=time.time()  

        ## load what is needed from keyvals. These are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.
        nx,ny,nw = keyvals[0:3]
        nline    = keyvals[3]
        tline    = keyvals[4]
        crpix3   = keyvals[7]         ## The wavelength domains will be different for each input line. nl keeps the index of the line to solve.
        crval3   = keyvals[10]
        cdelt3   = keyvals[13]
        
        ## needed data arrays
        specout  = np.zeros((nx,ny,nline,12),dtype=np.float32)         ## output array containing the spectroscopic products.
        wlarr    = np.zeros((nw,nline),dtype=np.float32)               ## wavelength array
        sobs_cal = np.zeros((nx,ny,nw,nline*4),dtype=np.float32)       ## define this as 0 and later check if the calibs are performed.
        ##NOTE: sobs_cal output should be in the same shape of the input array; e.g. [x,y,w,4*nline]

        ## create a wavelength array based on keywords for each line to be processed
        for i in prange(nline):
            if crpix3[i] == 0:                                         ## simple; just cycle and update
                for j in range(nw):
                    wlarr[j,i] = crval3[i]+(j*cdelt3[i])
            else:                                                      ## If reference pixel is not 0, cycle forwards and backwards from crpix3
                for j in range(crpix3[i],nw):                          ## forward cycle
                    wlarr[j,i] = crval3[i]+((j-crpix3[i])*cdelt3[i])
                for j in range(crpix3[i],-1,-1):                       ## Backwards cycle; -1 end index to fill the 0 index of the array
                    wlarr[j,i] = crval3[i]-((crpix3[i]-j)*cdelt3[i])

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

        if np.count_nonzero(sobs_cal) == 0:                            ## make sobs_cal from sobs_in
            if nline == 2:                                                             
                sobs_cal=np.append(sobs_in[0],sobs_in[1],axis=3).reshape(nx,ny,nw,8)
            else:
                sobs_cal=sobs_in[0]

        ######################################################################
        ## process the spectroscopy 
        if params.verbose >= 3:      ## Some print output          
            if nline == 2:
                for xx in range(nx):
                    print("SPECTRO_PROC: Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                    for yy in prange(ny):
                        print("                 Executing calculation: ",yy," of ",ny)
                        specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                            background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)    
                        specout[xx,yy,1,:] = cdf_statistics(sobs_cal[xx,yy,:,4:8],sobs_tot[xx,yy,4:8],\
                            background[xx,yy,4:8],wlarr[:,1],keyvals,consts.Constants(tline[1]),params.gaussfit,params.verbose)
            else:
                for xx in range(nx):
                    print("SPECTRO_PROC: Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                    for yy in prange(ny):
                        print("                 Executing calculation: ",yy," of ",ny)
                        specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                            background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)  

        elif params.verbose >= 1:    ## Some print output          
            if nline == 2:
                for xx in range(nx):
                    print("SPECTRO_PROC: Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                    for yy in prange(ny):
                        specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                            background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)    
                        specout[xx,yy,1,:] = cdf_statistics(sobs_cal[xx,yy,:,4:8],sobs_tot[xx,yy,4:8],\
                            background[xx,yy,4:8],wlarr[:,1],keyvals,consts.Constants(tline[1]),params.gaussfit,params.verbose)
            else:
                for xx in range(nx):
                    print("SPECTRO_PROC: Executing ext. loop: ",xx," of ",nx," (",ny," calculations / loop )")
                    for yy in prange(ny):
                        specout[xx,yy,0,:] = cdf_statistics(sobs_cal[xx,yy,:,0:4],sobs_tot[xx,yy,0:4],\
                            background[xx,yy,0:4],wlarr[:,0],keyvals,consts.Constants(tline[0]),params.gaussfit,params.verbose)  

        else:                        ##No print output
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

        ## NOTE: cdf_statistics will alter sobs_cal; It is not outputed or used downstream, so no issues should appear!

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
        if params.verbose >= 1: 
            if params.verbose >= 2: print("{:4.6f}".format(time.time()-start),' SECONDS FOR TOTAL SPECTROSCOPY PROCESSING')
            print('--------------------------------------\n-SPECTRO_PROC - SPECTROSCOPY FINALIZED\n--------------------------------------')

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
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache) ##object-mode because sps module is numba imcompatible
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
## fraction of total polarization

    #### NO OBSERVATION --> NO RUN! ###########    
    ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
    ##returns an array-like 0 vector of the dimensions of the requested output.
    if np.isnan(sobs_cal).all() or np.count_nonzero(sobs_cal) == 0:
        nullout = np.zeros((12),dtype=np.float32)
        if verbose >= 3: print("SPECTRO_PROC: FATAL! No observation in voxel!")
        return nullout
    
    ######################################################################    
    ## variable unpacks and preprocesses

    ## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.
    nw        = keyvals[2]
    cdelt3    = wlarr_1pix[1] - wlarr_1pix[0]          ## Line-agnostic cdelt3 computed locally. workaround to reduce the number of input vars.
    instwidth = keyvals[15]
    # needed arrays
    spec_1pix = np.zeros((12),dtype=np.float32)        ## output array containing the spectroscopic products.
    issuemask = np.zeros(5,dtype=np.int32)             ## temporary placeholder for issuemask
 
    ######################################################################
    ## Preprocess the spectral data
    
    ## NOTE: convoluted process.
    ## Noise reduce the spectral input data. This is technically a preprocess step.
    ## background subtraction using ackground and raw sobs_in arrays are processed here as part of cdf_statistics. 
    ## This is done this way because background is also desired as an output product, 
    
    #for i in range(4):                                       ## range(4); only 1 line is fed at a time to cdf_statistics
    #    sobs_cal[:,i]=sobs_cal[:,i]-background[i]
    sobs_cal -= background                                    ## faster multiplication

    ## now recompute the cdf for the noise-reduced data
    cdf       = obs_cdf(sobs_cal[:,0]) 

    ## check if there is reliable signal to fit and analyze
    ##(1) we can fit a line to the cdf distribution and in case it does fit well, there is no (reliable) stokes profile to recover.
    ##the 1.4e-5 rest in correlation corresponds to a gaussion with a peak in intensity of <5% compared to the average signal.
    if 1.- sps.pearsonr(wlarr_1pix,cdf)[0] < 1.4e-5:
        issuemask[0]=1
        if verbose >=3: print("SPECTRO_PROC: WARNING! No reliable Stokes I signal in pixel...")

    ## compute the center of the distribution, the line width, and the doppler shifts in wavelength and velocity.
    ## a normal distribution 0.5 is the average, sigma=34.13; FWHM=2*sqrt(2*alog(2))*sigma,
    ## [0] left stat fwhm, [1] center stat fwhm, [2] right stat fwhm;
    ## [3]left fwhm wave position, [4] fwhm center wave position, [5] fwhm right wave position;
    ## [6],[7],[8] just record the LEFT array indexes for recorded positions at [3], [4], [5];
    ##tmp = [0.5-2*np.sqrt(2*np.log(2))*34.13/2./100, 0.5, 0.5+2*np.sqrt(2*np.log(2))*34.13/2./100 ] ## /2/100 is the half width percentage.
    tmp = np.array((0.09815,0.5,0.90185,0,0,0,0,0,0),dtype=np.float32) 
    issuestr=["FWHM left margin not found","Line center not found","FWHM right margin not found"]
    for k in range(0,3):
        al       = np.argwhere( cdf[:-1] >= tmp[k] )[0,0]                        ## find the normed difference to theoretical centre from the k bin.             
        tmp[k+3] = wlarr_1pix[al]+cdelt3*2.*(tmp[k]-cdf[al])/(cdf[al+1]-cdf[al]) ##k1 written explicitly #k1=2*(tmp[k]-cdf[al])/(cdf[al+1]-cdf[al])
        tmp[k+6] = al
        if ((tmp[k+6] >= nw-2) or (tmp[k+6] == 0)):
            issuemask[k+2] = 1
            if verbose >=3: print("SPECTRO_PROC: WARNING! "+issuestr[j])
            
    ## fudge conversion for not doing repeated type conversions
    tmp6,tmp7,tmp8 = tmp[6:9].astype(np.int32)

    ## check if there is reliable signal to fit and analyze  (2) if the distribution is skewed, the inner part of it should not fit a line.
    if tmp8 - tmp6 >= 2: 
        if 1. - sps.pearsonr(wlarr_1pix[tmp6:tmp8+1],cdf[tmp6:tmp8+1])[0] > 5e-3:
            issuemask[1] = 1
            if verbose >= 3: print("SPECTRO_PROC: WARNING! Emission does not follow a normal distribution")
    else:                      ## not reliable signal == no products! 
        nullout = np.zeros((12),dtype=np.float32)
        if verbose >= 3: print("SPECTRO_PROC: FATAL! Spectroscopy not resolvable in pixel")
        return nullout            

    ######################################################################            
    ## The spectroscopic products for each line are computed here.   
    
    ## core wavelength and width
    if   gaussfit == 2:                                                                ## CDF + GAUSS version;
        gfit,gcov    = curve_fit(obs_gaussfit,wlarr_1pix,sobs_cal[:,0],p0=[np.max(sobs_cal[:,0]),tmp[4],(tmp[5]-tmp[3])/2.3548,0.0],maxfev=5000)
        spec_1pix[0] = gfit[1]                                                         ## line core wavelength
        spec_1pix[8] = 2.3548*gfit[2]                                                  ## Total line width [nm]; 2*np.sqrt(2*np.log(2))==2.3548
    elif gaussfit == 1:                                                                ## GAUSS version; still need cdf for extra calculations
        gfit,gcov    = curve_fit(obs_gaussfit,wlarr_1pix,sobs_cal[:,0],p0=[np.max(sobs_cal[:,0]),const.line_ref,0.2/2.3548,0.0],maxfev=5000)
        spec_1pix[0] = gfit[1]                                                         ## line core wavelength
        spec_1pix[8] = 2.3548*gfit[2]                                                  ## Total line width [nm]; 2*np.sqrt(2*np.log(2))==2.3548
    elif gaussfit == 0:                                                                ## CDF only version
        spec_1pix[0] = tmp[4]                                                          ## line core wavelength
        spec_1pix[8] = tmp[5]-tmp[3]                                                   ## Total line width [nm]

    ## Doppler shifts
    spec_1pix[1] = spec_1pix[0]-const.line_ref                                         ## line  shift from reference position [nm]
    spec_1pix[2] = spec_1pix[1]*const.l_speed*1e-3/const.line_ref                      ## shift in velocities; 1e-3 conversion from m/s to km/s
    
    ## record the intensity of the central wavelength for IQU profiles. 
    spec_1pix[3:6] = (sobs_cal[tmp7,0:3]+sobs_cal[tmp7+1,0:3])/2.                      ## Stokes IQU core intensity

    ## Stokes V Intensity. It will count the min/max counts of the first (left) lobe. will not match wavelength position of the other 3 quantities.
    a=np.argwhere(sobs_cal[:,3] == np.min(sobs_cal[:,3]))[0,0]                         ## location of the negative V lobe respective to line core.
    if a <= tmp7:
        spec_1pix[6] = (sobs_cal[a,3])                                                 ## is negative V lobe on the left? Stokes V has a -+ shape.
    else:
        b=np.argwhere(sobs_cal[:,3] == np.max(sobs_cal[:,3]))[0,0]
        spec_1pix[6] = (sobs_cal[b,3])                                                 ## Otherwise, Stokes V has a +- shape.

    ##background counts # Background should be similar in all four Stokes components
    spec_1pix[7] = background[0]                                                       ## background intensity of stokes I.
    
    ##line widths (non-thermal)  ## Non-thermal component of the line width that is dependent on instwidth # in order the lines are:
    ## check if width greater than what pure thermal broadening is (in addition to its voigt profile)   
    ## total line broadening  
    ## minus instrumental broadening
    ## minus thermal broadening 
    ## if width smaller; just set the non-thermal width to 0     
    spec_1pix[9] = spec_1pix[8] > 0.0975\
                    and np.sqrt( (((((spec_1pix[8]*1e-9)**2)*(const.l_speed**2))\
                    -(((instwidth*1e-9)**2)*((const.line_ref*1e-9)**2)))/(4*np.log(2)*((const.line_ref*1e-9)**2)))\
                    -(const.kb*(10.**const.ion_temp)/const.ion_mass))*const.line_ref/const.l_speed\
                    or 0                                                                                                
    ## if all other tests turn true, a line can still be "abnormally" narrow, especially in synthetic data at the limb
    ## We check using a logical test that the line is wider than the thermal width in order to compute the nt widths. 
    ## Otherwise the nt widths are set to 0.
    ## This is done to catch numerical errors where a log of negative numbers might appear.
    
    ## compute the polarization quantities
    spec_1pix[10] = np.sqrt(sobs_tot[1]**2+sobs_tot[2]**2)/sobs_tot[0]                 ## fraction of linear polarization with respect to intensity
    spec_1pix[11] = np.sqrt(sobs_tot[1]**2+sobs_tot[2]**2+sobs_tot[3]**2)/sobs_tot[0]  ## fraction of total polarization with respect to intensity
    
    return spec_1pix
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)     
def cledb_matchiquv(sobs_1pix,yobs_1pix,aobs_1pix,database_in,dbhdr,rms,nsearch,maxchisq,bcalc,reduced,verbose):
## main solver for the geometry and magnetic field strength version for full stokes vector
## Returns matched database index, double line IQUV vector, and chi^2 fitting residual
## Returns matched observation physics:
## Y(radial) position and X(LOS) position;
## varphi and vartheta angles in CLE geometry;
## Bx, By, Bx in LOS geometry (CLE transform to cartesian);
## B field strength.
    
    ## unpack dbcgrid parameters from the database accompanying db.hdr file 
    dbcgrid, ned, ngx, nbphi, nbtheta, xed, gxmin,gxmax, bphimin, bphimax, \
        bthetamin, bthetamax, nline, wavel  = dbhdr
    ####NO OBSERVATION --> NO RUN! ###########    
    ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
    ##returns an array-like 0 vector of the dimensions of the requested output.
    ## the index is set to -1 to warn about the missing entry
    if np.isnan(sobs_1pix).all() or np.count_nonzero(sobs_1pix) == 0:
        nullout = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 3: print("CLEDB_MATCHIQUV: FATAL! No observation in voxel!")
        return nullout,np.zeros((nsearch,8),dtype=np.float32)
    
    ##  Read the database in full or reduced mode ## database_sel becomes the reduced database in both cases
    if reduced == True:
        ## outredindex is the corresponding index for reduced, this is different from the full index. they need to be matched later.
        ## cledb_getsubset will reshape database_in to database_sel of [index,nline*4] shape
        database_sel,outredindex = cledb_getsubsetiquv(sobs_1pix,dbhdr,database_in,nsearch) 
        ## keeps the argument for the reduced entry using the sign of Stokes V. outargs+outredindex will return the correct initial db index below.
        outargs = np.argwhere(np.sign(database_sel[:,3]) == np.sign(sobs_1pix[3]))[:,0]   
        ## check and presort the sign along the first line Stokes V. Do not reverse order with above line to keep index mapping consistent!  
        database_sel = database_sel[outargs]                                              
    else:
        ## In this "else" branch, outredindex is not used, but numba does not properly compile because the array is presumably used for matching.
        outredindex = np.empty((nbphi,nsearch),dtype=np.float32)                          
        ## Reshape is found to be the fastest python/numpy/numba method to reorganize array dimensions.
        database_in = np.reshape(database_in,(ned*ngx*nbphi*nbtheta,8))                  
        ## keeps the argument for the reduced entry using the sign of Stokes V. outargs will return the correct initial db index below.
        outargs = np.argwhere(np.sign(database_in[:,3]) == np.sign(sobs_1pix[3]))[:,0]   
        ## check and presort the sign along the first line Stokes V. Do not reverse order with above line to keep index mapping consistent! 
        database_sel = database_in[outargs]                                               


    ## Normalize the input data to the strongest component and do a scaling to assign more equal weights to QU similar to I
    ## NOTE: Below we use these factors heavily to avoid altering the sobs_1pix or database_sel arrays 
    norm_fact       = sobs_1pix[np.argwhere(sobs_1pix == np.max(sobs_1pix))[0,0]]     ## normalization factor for the observation
    scale_fact      = np.ones(8,dtype=np.float32)                                     ## scaling factor for Q and U components
    scale_fact[1:7] = 10**(-np.floor(np.log10(np.abs(sobs_1pix[1:7]/norm_fact))) - 1) ## I1 and V2 have scale factors of 1
    scale_fact[3:5] = 1                                                               ## V1 and I2 also get scale factors of 1


    ## Geometric solution is based on a reduced chi^2 measure fit. We match sobs data with database_sel using the reduced chi^2 method.
    ## If observation has Stokes V, we include it in the difference for the chi^2, although it will have low influence because of small v/i.
    diff = np.zeros((database_sel.shape[0],8),dtype=np.float32)
    diff = (((database_sel[:,:8] - sobs_1pix/norm_fact)/rms)*scale_fact)  ## Multiply by the linear polarization scale factor to improve matches. 
 
    ndata = 4*2-1             ## Number of observables
    denom = ndata-4           ## Denominator used below in reduced chi^2      ## 4 = number of degree of freedom in model: ne, x, bphi, btheta
    diff *= diff              ## pure python multiplication was found to be faster than numpy or any power function applied.
    
    ## Unorthodox definition: chisq needs to be initialized separately, because np.round_ is not supported as a addressable function
    ## This is the only numba compatible implementation possible with current numba (0.51).
    ## Precision is up to 15 decimals; Significant truncation errors appear if this is not enforced. 
    ## if no rms and counts are present, the truncation is beyond a float32 without this decimal fix.    
    chisq = np.zeros((diff.shape[0]),dtype=np.float64)
    np.round_( np.sum( diff, axis=1)/denom/norm_fact,15,chisq)     
    ## Divide by norm_factor as the pure chi^2 is a number dependent metric. All variables are dependent on norm_fact => chisq = norm_fact*chi^2.

    ## Need to use a manual numba compatible fast sorting as np.argpartition is not numba implemented!
    ## cledb_partsort is a simple parallel function that produces an output similar to np.argpartition with sort.
    ## it does a standard sorting, aranging only the first nsearch elements, making it much faster than np.argsort for few (<100) solutions.

    if nsearch <= 100:
        asrt = cledb_partsort(chisq,nsearch) 
    else: ## for a lot of solutions, a full sort will perform better
        asrt =np.argsort(chisq)[0:nsearch]

    ## the following will compute the first nsearch solutions
    ## OR
    ## solutions that have chi^2 less than maxchisq 
    ## whichever comes first!!

    ## initialize return indices and chisq of nearest solutions
    si       = 0
    ix       = asrt[0]
    ixr      = asrt[0]
    ixrchisq = chisq[ixr]
    
    ## returns an array-like 0 vector of the dimensions of the requested output. The index is set to -1 to warn about the missing entry.
    if chisq[ixr] > maxchisq:                              ## no entries are compatible with the maxchisq requirement
        nullout      = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 3: print("CLEDB_MATCHIQUV: WARNING! No solutions in this pixel compatible with the maximum Chi^2 constraint.")
        return nullout,np.zeros((nsearch,8),dtype=np.float32) 
    
    ## arrays for storing the results
    out      = np.zeros((nsearch,11),dtype=np.float32)
    out[:,0] = np.full(nsearch,-1)                          ## start all indexes as not found, then update as you go
    smatch   = np.zeros((nsearch,8),dtype=np.float32)
    
    ## calculate the physics from the database and fill the output arrays
    for si in range(nsearch):
        
        ixr          = asrt[si]                  ## index in (presumably reduced) databasex
        ixrchisq     = chisq[ixr]                ## the chisq should correspond to the data array(ixr).
        if ixrchisq <= maxchisq:                 ## Are we still computing this si entry?

            ## Magnetic field strength from ratio of database with V/I ratios or wave data of observations and database.
            ## The bcalc estimation employs a logical test to avoid division by 0 in cases where the Zeeman signal vanishes due to geometry.
            ## If Stokes V is less than 1e-7, the matched field strength is 0 regardless of contents of the observation (usually it is very small).
            if bcalc == 0:                       ## using first line
                ## for division operation precision when database bfields are close to 0;
                bfield = np.abs(database_sel[ixr,3])>1e-7 and (sobs_1pix[3]/norm_fact) / database_sel[ixr,3] or 0 
                #bfield = (sobs_1pix[3]/norm_fact)/(database_sel[ixr,3]+1e-8) 
            if bcalc == 1:                       ## using second line
                bfield = np.abs(database_sel[ixr,7])>1e-7 and (sobs_1pix[7]/norm_fact) / database_sel[ixr,7] or 0
                #bfield = (sobs_1pix[7]/norm_fact)/(database_sel[ixr,7]+1e-8)
            if bcalc == 2:                       ## using the average of the two lines
                bfield = 0.5*((np.abs(database_sel[ixr,7])>1e-7 and (sobs_1pix[7]/norm_fact) / database_sel[ixr,7] or 0) +\
                              (np.abs(database_sel[ixr,3])>1e-7 and (sobs_1pix[3]/norm_fact) / database_sel[ixr,3] or 0))
                #bfield = 0.5*((sobs_1pix[3]/norm_fact)/(database_sel[ixr,3]+1e-8) + (sobs_1pix[7]/norm_fact)/(database_sel[ixr,7]+1e-8) )
            ## No bcalc == 3 in a full stokes inversion ## this is enforced upstream in cledb_invproc

            ## matching profiles to compare with the original observation
            smatch[si,:] = cledb_quderotate(database_sel[ixr,:],aobs_1pix,norm_fact)  ## ixr is the right index for database_sel
            
            # here if reduced we must get the original value of ix to write to output via outargs+outredindex.
            if reduced == True:
                ## first bring back the cut index via outargs, 
                ## then feed it alongside outredinxed via cledbparams+cledb_invparams to retrieve the initial ix index.
                ## ugly code. 0 list are just to reproduce the shape of dbcgrid
                i,j,k,l    = cledb_params(outargs[ixr],np.array((dbcgrid[0],dbcgrid[1],dbcgrid[2],outredindex.shape[1],0,0,0,0,0,0,0,0,0,0),dtype=np.float32))  
                ix         = cledb_invparams(i,j,k,np.int64(outredindex[k,l]),dbcgrid)  ## original value ix using the reduced outredindex to replace "l"
                ##update the inversion output array
                out[si,0]  = ix
                out[si,1]  = ixrchisq
                out[si,2:] = cledb_phys(ix,yobs_1pix,dbhdr,bfield)
            else:
                ix = asrt[si]                            
                ##update the inversion output array
                out[si,0]  = outargs[ix]                  ## uses outargs to get correct index to retrieve the physics
                out[si,1]  = ixrchisq
                out[si,2:] = cledb_phys(outargs[ix],yobs_1pix,dbhdr,bfield)

    ######################################################################
    ## [placeholder for issuemask]

    return out,smatch
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_matchiqud(sobs_1pix,sobsd_1pix,yobs_1pix,aobs_1pix,database_in,dbhdr,rms,nsearch,maxchisq,bcalc,reduced,verbose):
## main solver for the geometry and magnetic field strength version for partial stokes vector
## Returns matched database index, double line IQUV vector (for B=1G), and chi^2 fitting residual
## Returns matched observation physics:
## Y(radial) position and X(LOS) position;
## varphi and vartheta angles in CLE geometry;
## Bx, By, Bx in LOS geometry (CLE transform to cartesian);
## B field strength.
    
    ## unpack dbcgrid parameters from the database accompanying db.hdr file 
    dbcgrid, ned, ngx, nbphi, nbtheta, xed, gxmin,gxmax, bphimin, bphimax, \
        bthetamin, bthetamax, nline, wavel  = dbhdr
    
    #### NO OBSERVATION --> NO RUN! ###########    
    ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
    ##returns an array-like 0 vector of the dimensions of the requested output.
    ## the index is set to -1 to warn about the missing entry
    if np.isnan(sobs_1pix).all() or np.count_nonzero(sobs_1pix) == 0:
        nullout = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 3: print("CLEDB_MATCHIQUV: FATAL! No observation in voxel!")
        return nullout,np.zeros((nsearch,8),dtype=np.float32)
    
    ##  Read the database in full or reduced mode ## database_sel becomes the reduced database in both cases
    if reduced == True:
        ## outredindex is the corresponding index for reduced, this is different from the full index. they need to be matched later.
        ## cledb_getsubset will reshape database_in to database_sel of [index,nline*4] shape
        database_sel,outredindex = cledb_getsubsetiqud(sobs_1pix,sobsd_1pix,dbhdr,database_in,nsearch) 
        ###### Different from IQUV implementation ######    
        ## keeps the argument for the reduced entry based on the sign of B_pos; outargs+outredindex will return the correct initial db index below.
        ##outargs = np.argwhere(np.sign(database_sel[:,3]) == np.sign(sobsd_1pix[2]+1e-7))[:,0]   ## not yet validated
        outargs = np.argwhere(np.sign(database_sel[:,3]) == np.sign(database_sel[:,3]))[:,0]  ## crutch, will yeald 4 deg solutions
        ## check and presort the sign using doppler information. Do not reverse order with above line to keep index mapping consistent!  
        database_sel = database_sel[outargs]                                              
    else:
        ## In this "else" branch, outredindex is not used, but numba does not properly compile because the array is presumably used for matching.
        outredindex = np.empty((nbphi,nsearch),dtype=np.float32)                          
        ## Reshape is found to be the fastest python/numpy/numba method to reorganize array dimensions.
        database_in = np.reshape(database_in,(ned*ngx*nbphi*nbtheta,8))                  
        ###### Different from IQUV implementation ###### 
        ## keeps the argument for the reduced entry using the sign of B_pos. outargs will return the correct initial db index below.
        #outargs=np.argwhere(np.sign(database_in[:,3]) == np.sign(sobsd_1pix[2]+1e-7))[:,0] ##not yet validated
        outargs=np.argwhere(np.sign(database_in[:,3]) == np.sign(database_in[:,3]))[:,0]  ## crutch, will yeald 4 deg solutions
        ## check and presort the sign using doppler information. Do not reverse order with above line to keep index mapping consistent! 
        database_sel = database_in[outargs]                                               


    ## Normalize the input data to the strongest component and do a scaling to assign more equal weights to QU similar to I
    ## NOTE: Below we use these factors heavily to avoid altering the sobs_1pix or database_sel arrays 
    norm_fact       = sobs_1pix[np.argwhere(sobs_1pix == np.max(sobs_1pix))[0,0]]     ## normalization factor for the observation
    scale_fact      = np.ones(8,dtype=np.float32)                                     ## scaling factor for Q and U components
    scale_fact[1:7] = 10**(-np.floor(np.log10(np.abs(sobs_1pix[1:7]/norm_fact))) - 1) ## I1 and V2 have scale factors of 1
    scale_fact[3:5] = 1                                                               ## V1 and I2 also get scale factors of 1


    ## Geometric solution is based on a reduced chi^2 measure fit. We match sobs data with database_sel using the reduced chi^2 method.
    ## If observation has Stokes V, we include it in the difference for the chi^2, although it will have low influence because of small v/i.
    ## We can recover the geometry without using Stokes V. An additional degeneration manifests. This is avoided above via the outargs definition
    
    diff = np.zeros((database_sel.shape[0],8),dtype=np.float32)
    diff = (((database_sel[:,:8] - sobs_1pix/norm_fact)/rms)*scale_fact)  ## Multiply by the linear polarization scale factor to improve matches. 
 
    ndata = 3*2-1             ## Number of observables ## IQUD has only 6 entry opservations instead of 8
    denom = ndata-4           ## Denominator used below in reduced chi^2      ## 4 = number of degree of freedom in model: ne, x, bphi, btheta
    diff *= diff              ## pure python multiplication was found to be faster than numpy or any power function applied.
    
    ## Unorthodox definition: chisq needs to be initialized separately, because np.round_ is not supported as a addressable function
    ## This is the only numba compatible implementation possible with current numba (0.51).
    ## Precision is up to 15 decimals; Significant truncation errors appear if this is not enforced. 
    ## if no rms and counts are present, the truncation is beyond a float32 without this decimal fix.    
    chisq = np.zeros((diff.shape[0]),dtype=np.float64)
    np.round_( np.sum( diff, axis=1)/denom/norm_fact,15,chisq)     
    ## Divide by norm_factor as the pure chi^2 is a number dependent metric. All variables are dependent on norm_fact => chisq = norm_fact*chi^2.

    ## Need to use a manual numba compatible fast sorting as np.argpartition is not numba implemented!
    ## cledb_partsort is a simple parallel function that produces an output similar to np.argpartition with sort.
    ## it does a standard sorting, aranging only the first nsearch elements, making it much faster than np.argsort for few (<100) solutions.

    if nsearch <= 100:
        asrt = cledb_partsort(chisq,nsearch) 
    else: ## for a lot of solutions, a full sort will perform better
        asrt =np.argsort(chisq)[0:nsearch]

    ## the following will compute the first nsearch solutions
    ## OR
    ## solutions that have chi^2 less than maxchisq 
    ## whichever comes first!!

    ## initialize return indices and chisq of nearest solutions
    si       = 0
    ix       = asrt[0]
    ixr      = asrt[0]
    ixrchisq = chisq[ixr]
    
    ## returns an array-like 0 vector of the dimensions of the requested output. The index is set to -1 to warn about the missing entry.
    if chisq[ixr] > maxchisq:                              ## no entries are compatible with the maxchisq requirement
        nullout      = np.zeros((nsearch,11),dtype=np.float32)
        nullout[:,0] = np.full(nsearch,-1)
        if verbose >= 3: print("CLEDB_MATCHIQUV: WARNING! No solutions in this pixel compatible with the maximum Chi^2 constraint.")
        return nullout,np.zeros((nsearch,8),dtype=np.float32) 
    
    ## arrays for storing the results
    out      = np.zeros((nsearch,11),dtype=np.float32)
    out[:,0] = np.full(nsearch,-1)                          ## start all indexes as not found, then update as you go
    smatch   = np.zeros((nsearch,8),dtype=np.float32)       ## we preserve smatch with 8 variables to record the stokes v in the database
    
    ## calculate the physics from the database and fill the output arrays
    for si in range(nsearch):
        
        ixr          = asrt[si]                  ## index in (presumably reduced) databasex
        ixrchisq     = chisq[ixr]                ## the chisq should correspond to the data array(ixr).
        if ixrchisq <= maxchisq:                 ## Are we still computing this si entry?

            ###### Different from IQUV implementation ###### 
            ## Magnetic field strength from wave data of observations.
            ## No bcalc == 1 or bcalc== 2 in a iqud inversion ## This is enforced in the upstream cledb_invproc function
            if bcalc == 3: ## using the fieldstrength from the wave tracking
                bfield = sobsd_1pix[0]

            ## matching profiles to compare with the original observation
            smatch[si,:] = cledb_quderotate(database_sel[ixr,:],aobs_1pix,norm_fact)  ## ixr is the right index for database_sel
            
            # here if reduced we must get the original value of ix to write to output via outargs+outredindex.
            if reduced == True:
                ## first bring back the cut index via outargs, 
                ## then feed it alongside outredinxed via cledbparams+cledb_invparams to retrieve the initial ix index.
                ## ugly code. 0 list are just to reproduce the shape of dbcgrid
                i,j,k,l    = cledb_params(outargs[ixr],np.array((dbcgrid[0],dbcgrid[1],dbcgrid[2],outredindex.shape[1],0,0,0,0,0,0,0,0,0,0),dtype=np.float32))  
                ix         = cledb_invparams(i,j,k,np.int64(outredindex[k,l]),dbcgrid)  ## original value ix using the reduced outredindex to replace "l"
                ##update the inversion output array
                out[si,0]  = ix
                out[si,1]  = ixrchisq
                out[si,2:] = cledb_phys(ix,yobs_1pix,dbhdr,bfield)
            else:
                ix = asrt[si]                            
                ##update the inversion output array
                out[si,0]  = outargs[ix]                  ## uses outargs to get correct index to retrieve the physics
                out[si,1]  = ixrchisq
                out[si,2:] = cledb_phys(outargs[ix],yobs_1pix,dbhdr,bfield)

    ######################################################################
    ## [placeholder for issuemask]

    return out,smatch
###########################################################################
###########################################################################


# ###########################################################################
# ###########################################################################
# @njit(parallel=params.jitparallel,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
# def cledb_matchiqud(sobs_1pix,sobsd_1pix,yobs_1pix,aobs_1pix,database_sel,dbhdr,rms,nsearch,maxchisq,bcalc,reduced,verbose):
# ## main solver for the geometry and magnetic field strength -- Version with linear polarization+doppler data.
# ## Returns matched database index, double line IQU vector, and chi^2 fitting residual
# ## Returns matched observation physics:
# ## Y(radial) position and X(LOS) position;
# ## B_phi and B_theta angles in L.V.S geometry;
# ## Bx, By, Bx in LOS geometry (L.V.S. transform to cartesian);
# ## B field strength.

#     ## unpack dbcgrid parameters from the database accompanying db.hdr file 
#     dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
#         bthetamin, bthetamax, nline, wavel  = dbhdr
#     ####NO OBSERVATION --> NO RUN! ###########    
#     ## e.g. a pixel inside the solar disk, an invalid pixel, etc.
#     ##returns an array-like 0 vector of the dimensions of the requested output.
#     ## the index is set to -1 to warn about the missing entry
#     if np.isnan(sobs_1pix).all() or np.count_nonzero(sobs_1pix) == 0:
#         nullout = np.zeros((nsearch,11),dtype=np.float32)
#         nullout[:,0] = np.full(nsearch,-1)
#         if verbose >= 2: print("WARNING (CLEDB_MATCH): No observation in voxel!")
#         return nullout,np.zeros((nsearch,8),dtype=np.float32)

#     ##  Read the database in full or reduced mode  
#     if reduced == True:
#         ## database_sel becomes the reduced database in this case
#         ## outredindex is the corresponding index for reduced, this is different from the full index. they need to be matched later.
#         ## cledb_getsubset will reshape database_sel to [index,nline*4] shape
#         database_sel,outredindex = cledb_getsubset(sobs_1pix,dbhdr,database_sel,nsearch) 
#         if verbose >= 3:print("CLEDB_MATCH: using reduced DB:",database_sel.shape,outredindex.shape)
        
#     else:
#         outredindex = np.empty((nbphi,nsearch),dtype=np.float32)                ## In this branch, outredindex is never used, but numba does not properly compile because the array is presumably presumably used for matching.
#         database_sel = np.reshape(database_sel,(ned*ngx*nbphi*nbtheta,8))       ## Reshape is found to be the fastest python/numpy/numba method to reorganize array dimensions.
#         ## Different from IQUV implementation
#         outargs=np.argwhere(np.sign(database_sel[:,3]) == np.sign(sobsd_1pix[2]+1e-7) )[:,0]   ## keeps the argument for the reduced entry based on the sign from doppler waves.
#         database_sel = database_sel[np.sign(database_sel[:,3]) == np.sign(sobsd_1pix[2]+1e-8)]  ## check the sign of the stokes observation used to compute blos


#         if verbose >= 2:
#             if (database_sel.shape[0] > 1000000):
#                 print("WARNING (CLEDB_MATCH): Full database match will be significantly slower with no proven benefits!")
#             if verbose >= 3:print("CLEDB_MATCH: using full DB:",database_sel.shape,outredindex.shape)

#     ##normalize the input data to the strongest component
#     norm_fact=sobs_1pix[np.argwhere(sobs_1pix == np.max(sobs_1pix))[0,0]]
#     sobs_1pix[:]=sobs_1pix[:]/norm_fact
        
#     ## Geometric solution is based on a reduced chi^2 measure fit.
#     ## match sobs data with database_sel using the reduced chi^2 method                                                
#     ## Different from IQUV implementation
#     ## We do not use Stokes V to define the geometry. Therefore we use a subset (1,2,4,5,6) 
#     ## of Stokes parameters QU (line 1) and IQU (line 2); total=5
#     diff = np.zeros((database_sel.shape[0],6),dtype=np.float32)
#     ## These two below lines are the slowest of this function due to requiring two operations.
#     ## A better numba COMPATIBLE alternative was not found
#     diff[:,0:2] = (database_sel[:,1:3] - sobs_1pix[1:3]) /rms[1:3]#/ np.abs(rms[1:3])
#     diff[:,2:5] = (database_sel[:,4:7] - sobs_1pix[4:7]) /rms[4:7]#/ np.abs(rms[4:7])
    
    
#     ndata = 4*2-1-2          ## Number of observables for geometry solver ## ndata = -2 comes from not using the two V components (missing in IQUD).    
#     denom = ndata-4           ## Denominator used below in reduced chi^2      ## 4 = number of degree of freedom in model: ne, x, bphi, btheta
#     diff *= diff              ## pure python multiplication was found to be faster than numpy or any power function applied.
#     chisq = np.zeros((diff.shape[0]),dtype=np.float64)
#     np.round_( np.sum( diff, axis=1)/denom,15,chisq)
#     ## Unorthodox definition: chisq needs to be initialized separately, because np.round_ is not supported as a addressable function
#     ## This is the only numba compatible implementation possible with current numba implementation.
#     ## Precision is up to 15 decimals; Significant truncation errors appear if this is not enforced. 
#     ## if no rms and counts are present, the truncation is beyond a float32 without this decimal fix.

#     ## Need to use a manual numba compatible fast sorting as np.argpartition is not numba implemented!
#     ## cledb_partsort is a simple parallel function that produces an output similar to np.argpartition with sort.
#     ## it just does a standard sorting search, but sorts just the first nsearch elements, making it much faster than np.argsort for few solutions.
#     ## for a lot of solutions, a full sort will perform better
#     if nsearch <= 100:
#         asrt = cledb_partsort(chisq,nsearch) 
#     else:
#         if verbose >= 2: print("WARNING (CLEDB_MATCH): High number of solutions requested. Expect slow runtime. Using a full array sort!")
#         asrt =np.argsort(chisq)[0:nsearch]
    
#     ## the for loop will compute the first nsearch solutions;
#     ## OR
#     ## solutions that have chi^2 less than maxchisq; 
#     ## whichever comes first!!

#     ## initialize return indices and chisq of nearest solutions
#     si = 0
#     ix = asrt[0]
#     ixr = asrt[0]
#     ixrchisq = chisq[ixr]

#     ## returns an array-like 0 vector of the dimensions of the requested output.
#     ## the index is set to -1 to warn about the missing entry
#     if chisq[ixr] > maxchisq:
#         nullout = np.zeros((nsearch,11),dtype=np.float32)
#         nullout[:,0] = np.full(nsearch,-1)
#         if verbose >= 2: print("WARNING (CLEDB_MATCH): No solutions compatible with the maximum Chi^2 constraint.")
#         return nullout,np.zeros((nsearch,8),dtype=np.float32) 

#     ## arrays for storing the results
#     out=np.zeros((nsearch,11),dtype=np.float32)
#     out[:,0]=np.full(nsearch,-1)                  ## start all indexes as not found, then update as you go
#     smatch=np.zeros((nsearch,8),dtype=np.float32)

#     ## calculate the physics from the database and fill the output arrays
#     for si in range(nsearch):

#         ixr = asrt[si]                    ## index in (presumably reduced) databasex
#         ixrchisq = chisq[ixr]             ## the chisq should correspond to the data array(ixr), be it reduced or not
#         if ixrchisq <= maxchisq:          ## Are we still computing this si entry?

#             # Magnetic field strength from ratio of database with sign from wave tracking
#             if bcalc == 3: ## using the fieldstrength from the wave tracking
#                 bfield = sobsd_1pix[0]*np.sign(sobsd_1pix[2]+1e-8)

#             ## matching profiles to compare with the original observation
#             smatch[si,:] = database_sel[ixr,:]                                              ## if database is reduced ixr is the right index; otherwise ixr=ix
#             smatch[si,:] = cledb_quderotate(smatch[si,:],aobs_1pix,norm_fact)

#             # here if reduced we must get the original value of ix  to write to output (if required!).
#             if reduced == True:
#                 i,j,k,l = cledb_params(ixr,np.array((dbcgrid[0],dbcgrid[1],dbcgrid[2],outredindex.shape[1],0,0,0,0,0,0,0,0,0,0),dtype=np.float32))   ## extremely ugly but fast. 0 are just to reproduce the shape of dbcgrid
#                 ix = cledb_invparams(i,j,k,np.int64(outredindex[k,l]),dbcgrid)                                                      ## = original value ix using the reduced outredindex to replace "l"
#                 ##update the inversion output array
#                 out[si,0]=ix
#                 out[si,1]=ixrchisq
#                 out[si,2:]=cledb_phys(ix,yobs_1pix,dbhdr,bfield)
#             else:
#                 ix = asrt[si]                     ## this is updated if reduced is used
#                 ##update the inversion output array
#                 out[si,0]=outargs[ix]                               ### uses outargs to produce the correct physics
#                 out[si,1]=ixrchisq
#                 out[si,2:]=cledb_phys(outargs[ix],yobs_1pix,dbhdr,bfield)

#     ######################################################################
#     ## [placeholder for issuemask]

#     return out,smatch
# ###########################################################################
# ###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_getsubsetiquv(sobs_1pix,dbhdr,database_in,nsearch): 
## returns a subset of the sdb array compatible with the height in yobs

    ## unpack dbcgrid parameters from the database accompanying db.hdr file
    dbcgrid, ned, ngx, nbphi, nbtheta, xed, gxmin, gxmax, bphimin, bphimax,\
    bthetamin, bthetamax, nline, wavel = dbhdr 

    ## Create the reduced arrays for analysis; we don't need more than the desired nsearch subsets
    datasel     = np.zeros((ned,ngx,nbphi,2*nsearch,8),dtype=np.float32)
    outredindex = np.zeros((nbphi,2*nsearch),dtype=np.float32)
    ##NOTE: the sorting done here only takes into account the linear polarization tangent of the observation. 
    ##      This sorting is not 1:1 equivalent with the main chi^2 sorting.
    ##      To be sure all compatible solutions are captured nsearch*2 solutions must be enforced here, and further refined in cledb_match

    ## database: indexes of separate the CLE array calculations for reduction
    ## Don't use nbphi -1 or nbtheta-1 to avoid adding additional degeneracy at 0--pi or 0--2pi
    kk      = np.arange(0,nbphi)
    bphir   = bphimin + kk*(bphimax-bphimin)/(nbphi)           ## cle phi array 

    ll      = np.arange(0,nbtheta)
    bthetar = bthetamin + ll*(bthetamax-bthetamin)/(nbtheta)   ## cle theta array
    tt      = np.tan(bthetar)                                  ## cle tan theta array

    ## Observation: compute the PHI_B and its degenerate tangents
    phib_obs      = -0.5*np.arctan2(sobs_1pix[2],sobs_1pix[1]) ## tan Phi_B = sin phi * tan theta
    tphib_obs     = np.tan(phib_obs)                           ## here is tan Phi_B
    tphib_obs_deg = np.tan(phib_obs+np.pi/2.)                  ## and its degenerate branch

    # Find those indices compatible with phib observed
    for ir in range(bphir.shape[0]):                           ## loop over bphi in cle frame
        ttp   = tt * np.sin(bphir[ir])
        diffa = np.abs(tphib_obs - ttp)                        ## this is an array over btheta at each bphi
        diffb = np.abs(tphib_obs_deg - ttp)                    ## this is an array over btheta at each bphi (degenerate branch)
        srta  = cledb_partsort(diffa,nsearch)                  ## NOTE: no SIGNIFICANT speed gain to use PARTSORT here as the arrays are small.
        srtb  = cledb_partsort(diffb,nsearch)
        #srta=np.argsort(diffa)[0:nsearch]
        #srtb=np.argsort(diffb)[0:nsearch]
        ## advanced slicing is not available, the for jj enumeration comes from numba requirements
        if ir + srta[0] > 0 or ir + srtb[0] > 0:                                  ## important to avoid phi=0 AND theta = 0 case
            for jj in range(nsearch):                                             ## NOTE: nsearch = srt.shape[0]
                datasel[:,:,ir,jj,:]           = np.copy(database_in[:,:,ir,srta[jj],:])   ## Record those indices compatible with phib observed (main branch)
                outredindex[ir,jj]             = srta[jj]
                datasel[:,:,ir,jj+(nsearch),:] = np.copy(database_in[:,:,ir,srtb[jj],:])   ## Record those indices compatible with phib observed (deg. branch)
                outredindex[ir,jj+(nsearch)]   = srtb[jj]

    ## Note: the above block is both faster than concatenating the diff arrays + sorting only once + one subscription for datasel & outredindex.

    ## now work with reduced dataset with nbtheta replaced by nsearch
    ## the full dimensions of the database are no longer needed. It is converted to a [index,8] shape
    return np.reshape(datasel,(ned*ngx*nbphi*2*nsearch,8)),outredindex
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_getsubsetiqud(sobs_1pix,sobsd_1pix,dbhdr,database_in,nsearch): 
## returns a subset of the sdb array compatible with the height in yobs

    ## unpack dbcgrid parameters from the database accompanying db.hdr file
    dbcgrid, ned, ngx, nbphi, nbtheta, xed, gxmin, gxmax, bphimin, bphimax,\
    bthetamin, bthetamax, nline, wavel = dbhdr 

    ## Create the reduced arrays for analysis; we don't need more than the desired nsearch subsets
    datasel     = np.zeros((ned,ngx,nbphi,2*nsearch,8),dtype=np.float32)
    outredindex = np.zeros((nbphi,2*nsearch),dtype=np.float32)
    ##NOTE: the sorting done here only takes into account the linear polarization tangent of the observation. 
    ##      This sorting is not 1:1 equivalent with the main chi^2 sorting.
    ##      To be sure all compatible solutions are captured nsearch*2 solutions must be enforced here, and further refined in cledb_match

    ## database: indexes of separate the CLE array calculations for reduction
    ## Don't use nbphi -1 or nbtheta-1 to avoid adding additional degeneracy at 0--pi or 0--2pi
    kk      = np.arange(0,nbphi)
    bphir   = bphimin + kk*(bphimax-bphimin)/(nbphi)           ## cle phi array 

    ll      = np.arange(0,nbtheta)
    bthetar = bthetamin + ll*(bthetamax-bthetamin)/(nbtheta)   ## cle theta array
    tt      = np.tan(bthetar)                                  ## cle tan theta array

    ## Observation: compute the PHI_B and its degenerate tangents
    ## tan Phi_B = sin phi * tan theta
    ##phib_obs      = -0.5*np.arctan2(sobs_1pix[2],sobs_1pix[1]) ## Disabled
    phib_obs      = sobsd_1pix[1]                              ## using the wave phase angle instead of polarization arctangent
    tphib_obs     = np.tan(phib_obs)                           ## here is tan Phi_B
    tphib_obs_deg = np.tan(phib_obs+np.pi/2.)                  ## and its degenerate branch

    # Find those indices compatible with phib observed
    for ir in range(bphir.shape[0]):                           ## loop over bphi in cle frame
        ttp   = tt * np.sin(bphir[ir])
        diffa = np.abs(tphib_obs - ttp)                        ## this is an array over btheta at each bphi
        diffb = np.abs(tphib_obs_deg - ttp)                    ## this is an array over btheta at each bphi (degenerate branch)
        srta  = cledb_partsort(diffa,nsearch)                  ## NOTE: no SIGNIFICANT speed gain to use PARTSORT here as the arrays are small.
        srtb  = cledb_partsort(diffb,nsearch)
        #srta=np.argsort(diffa)[0:nsearch]
        #srtb=np.argsort(diffb)[0:nsearch]
        ## advanced slicing is not available, the for jj enumeration comes from numba requirements
        if ir + srta[0] > 0 or ir + srtb[0] > 0:                                  ## important to avoid phi=0 AND theta = 0 case
            for jj in range(nsearch):                                             ## NOTE: nsearch = srt.shape[0]
                datasel[:,:,ir,jj,:]           = np.copy(database_in[:,:,ir,srta[jj],:])   ## Record those indices compatible with phib observed (main branch)
                outredindex[ir,jj]             = srta[jj]
                datasel[:,:,ir,jj+(nsearch),:] = np.copy(database_in[:,:,ir,srtb[jj],:])   ## Record those indices compatible with phib observed (deg. branch)
                outredindex[ir,jj+(nsearch)]   = srtb[jj]

    ## Note: the above block is both faster than concatenating the diff arrays + sorting only once + one subscription for datasel & outredindex.

    ## now work with reduced dataset with nbtheta replaced by nsearch
    ## the full dimensions of the database are no longer needed. It is converted to a [index,8] shape
    return np.reshape(datasel,(ned*ngx*nbphi*2*nsearch,8)),outredindex
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_quderotate(dbarr,ang,nf):
## derotates the matched database entry to make it compatible with the observation; 
## same Mueller transform as obs_qurotate, but applied with the oposite angle to just want to compare the DB stokes profiles with the observed ones.

    ## use dbarr for updating as to not use modified arr[1] and arr[5] values for arr[2] and arr[6]
    arr=np.copy(dbarr)*nf ## don't alter exterior arrays 
    arr[1] = dbarr[1]*nf*np.cos(-2.*ang) - dbarr[2]*nf*np.sin(-2.*ang)
    arr[2] = dbarr[1]*nf*np.sin(-2.*ang) + dbarr[2]*nf*np.cos(-2.*ang)
    arr[5] = dbarr[5]*nf*np.cos(-2.*ang) - dbarr[6]*nf*np.sin(-2.*ang)
    arr[6] = dbarr[5]*nf*np.sin(-2.*ang) + dbarr[6]*nf*np.cos(-2.*ang)

    return arr
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_partsort(arr,nsearch):
## cledb_partsort is a function that produces an output similar to np.argpartition with sort.
## Simplest possible substitution partial sort. It just returns the first nsearch sorted indexes. The rest of the array is unsorted.
## It is fully numba non-python compatible!
   
    ## updated in update-iqud tag; original implementation is still commented
    # asrt=np.zeros((nsearch),dtype=np.int64)
    # arr_temp=np.copy(arr) ## don't change the input array inside; numba will complain of changing an upstream array that you don't return back.
    # for i in range (nsearch):
    #     a=np.argmin(arr_temp[i:])
    #     b=np.argwhere(arr == arr_temp[i:][a])[0]
    #     d=0
    #     for j in range(b.shape[0]):
    #         if ((b[j] in asrt[:i]) and (d+1 <b.shape[0])):
    #             d+=1
    #     asrt[i]=b[d]
    #     sort_temp=arr_temp[i]
    #     arr_temp[i]=arr_temp[i:][a]
    #     arr_temp[i:][a]=sort_temp
        
    asrt=np.zeros((nsearch),dtype=np.int64)
    arr_temp=np.copy(arr)            ## don't change the input array inside.
    ## The sorting puts the previously found value at the beginning of the array, then searches though a range excluding previously moved elements.
    for i in range (nsearch):
        a               = np.argmin(arr_temp[i:])
        asrt[i]         = a+i                  ## each i index iteration is corrected by +i
        sort_temp       = arr_temp[i]          ## move the foundindex to the i-th position
        arr_temp[i]     = arr_temp[i:][a]
        arr_temp[i:][a] = sort_temp
    
    return asrt         ##works like a charm!
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_params(index,dbcgrid):
## This function is used with the database header to help compute the physics associated to the i, j, k, l entry
## for index, get i,j,k,l indices in a database

    ned     = np.int32(dbcgrid[0])   ## upstream dbcgrid values should be of type np.int32. no type-casting should be required
    ngx     = np.int32(dbcgrid[1])
    nbphi   = np.int32(dbcgrid[2])
    nbtheta = np.int32(dbcgrid[3])
    n5      = ngx*nbphi*nbtheta
    n4      =     nbphi*nbtheta
    n3      =           nbtheta

    i = index   //        n5
    j = index   -     i * n5
    j = j      //         n4
    k = index   -     i * n5 - j * n4
    k = k      //         n3
    l = index   -     i * n5 - j * n4 - k * n3

    return i,j,k,l
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_invparams(i,j,k,l,dbcgrid):
## Reverse function of cledb_params
## This function is used with the database header to help compute the physics associated to the index entry
## for i,j,k,l in database, get index
    
    ## Description of whats in each dbcgrid used here
    #ned     = np.int32(dbcgrid[0]) 
    #ngx     = np.int32(dbcgrid[1])
    #nbphi   = np.int32(dbcgrid[2])
    #nbtheta = np.int32(dbcgrid[3])
    #return np.int32(i*ngx*nbphi*nbtheta + j*nbphi*nbtheta + k*nbtheta + l)

    return np.int32(i*dbcgrid[1]*dbcgrid[2]*dbcgrid[3] + j*dbcgrid[2]*dbcgrid[3] + k*dbcgrid[3] + l)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down
def cledb_elecdens(r):
## computes an electron radial density estimation using the Baumbach formulation

    baumbach = 1.e8*(0.036/r**1.5 + 1.55/r**6.)
    hscale   = 7.18401074e-02  # 50 Mm

    return np.float32(3.e8*np.exp(- (r-1.)/hscale) + baumbach)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down 
def cledb_phys(index,gy,dbhdr,b):
## Returns the CLE and Obs. geometry and magnetic field physics
## this is kept separate from cledb_physcle because it computes projections transformations of the database variables recovered via cledb_physcle.

    phs    = cledb_physcle(index,gy,dbhdr)
    bphi   = phs[3]
    btheta = phs[4]
    bx     = np.abs(b)*np.sin(btheta)*np.cos(bphi)                 ## The np.abs(b) comes from the standard spherical transform formalism. 
    by     = np.abs(b)*np.sin(btheta)*np.sin(bphi)
    bz     = np.abs(b)*np.cos(btheta)

    return np.array((phs[0],phs[1],phs[2],b,bphi,btheta,bx,by,bz),dtype=np.float32)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)      ## don't try to parallelize things that don't need as the overhead will slow everything down 
def cledb_physcle(index,gy,dbhdr):
## Helper for phys_par; returns the CLE frame physics and compatible observation geometry.
## this is the primary function that retrieves physics parameters from the database index.

    dbcgrid, ned, ngx, nbphi, nbtheta, xed, gxmin,gxmax, bphimin, bphimax,\
    bthetamin, bthetamax, nline, wavel  = dbhdr

    i,j,k,l = cledb_params(index,dbcgrid)

    gx      = gxmin + j*(gxmax-gxmin)/(ngx-1)                          ## ngx & ngy have -1 intervals and +1 points to traverse 0 and domain ends.
    bphi    = bphimin + k*(bphimax-bphimin)/(nbphi)                    ## nbphi & nbtheta dont have -1 to not double the solutions for 0&pi or 0&2pi.
    btheta  = bthetamin + l*(bthetamax-bthetamin)/(nbtheta)
    ed      = np.float32(xed[i]* cledb_elecdens(np.sqrt(gy*gy+gx*gx))) ## log of Ne

    return np.array((np.log10(ed),gy,gx,bphi,btheta),dtype=np.float32)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_cdf(spectr):
## computes the cdf distribution for a spectra

    cdf = np.zeros((spectr.shape[0]),dtype=np.float32)
    for i in prange (0,spectr.shape[0]):
        cdf[i] = np.sum(spectr[0:i+1])                        
    return cdf/cdf[-1]                      ## norm the cdf to simplify interpretation
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

###########################################################################
###########################################################################
def cledb_invproc_time(sobs_totrot,sobs_dopp,database,db_enc,yobs,aobs,rms,dbhdr,keyvals,nsearch,maxchisq,bcalc,iqud,reduced,verbose):
## a minimal wrapper that enables timing for CLEDB_INVERT. Timing can no be directly included in function due to full non-python numba compilation.
  
    if verbose >= 2: start0=time.time()  
 
    invout,sfound=cledb_invproc(sobs_totrot,sobs_dopp,database,db_enc,yobs,aobs,rms,dbhdr,keyvals,nsearch,maxchisq,bcalc,iqud,reduced,verbose)
    
    if verbose >= 2: print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL DB INVERSION PROCESSING')
  
    return invout,sfound
###########################################################################
###########################################################################
