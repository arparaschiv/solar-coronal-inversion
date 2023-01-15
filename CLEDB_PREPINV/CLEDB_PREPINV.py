# -*- coding: utf-8 -*-

###############################################################################
### Parallel implementation of utilities for CLEDB_PREPINV                  ###
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

### Other used packages of secondary importance that will probably not become code-breaking 
# pylab    ## Check if still needed
# numexpr
# glob

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
## Full Parallelization will generally not be possible when time functions are enabled via high verbosity.



#########################################################################
### Needed imports ######################################################
#########################################################################

#
# Needed libraries
#
import numpy as np

from numba import jit,njit, prange
from numba.typed import List  ## numba list is needed ad standard reflected python lists will be deprecated in numba

from scipy.io import FortranFile
from pylab import *
import time
import glob
import os
import sys
import numexpr as ne

import ctrlparams 
params=ctrlparams.ctrlparams()    ## just a shorter label

#########################################################################
### Main Modules ########################################################
#########################################################################

###########################################################################
###########################################################################
#@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)  ### numba incompatible due to obs_headstructproc
def sobs_preprocess(sobs_in,headkeys,params):
## Main script to read data and header information to prepare it for analysis.

    if params.verbose >=1: 
        print('------------------------------------\n--------OBS PROCESSING START--------\n------------------------------------')
        if params.verbose >= 3:start0=time.time()        

## unpack the minimal number of header keywords
    keyvals=obs_headstructproc(sobs_in,headkeys,params)
    if(keyvals == -1):
        if params.verbose >=1: 
            print('------------------------------------\n-----OBS KEYWORD PROCESSING FAIL----\n------------------------------------')
        return -1
    
##calculate an observation height and integrate the stokes profiles
    yobs=obs_calcheight(keyvals)
    if params.integrated != True:
        sobs_tot,rms,background=obs_integrate(sobs_in,keyvals)
        
    else:   ## set the arrays to -1 because they can't be computed
        sobs_tot   = np.concatenate((sobs_in[0],sobs_in[1]),axis=2)
        rms        = np.zeros((keyvals[0],keyvals[1],keyvals[3]*4),dtype=np.float32)-1
        background = np.zeros((keyvals[0],keyvals[1],keyvals[3]*4),dtype=np.float32)-1

    if keyvals[3] == 2:
    ## for two-line inversions we require a rotation of the linear polarization Q and U components
        sobs_totrot = np.copy(sobs_tot)
        aobs        = np.zeros((keyvals[0],keyvals[1]),dtype=np.float32)
        
        sobs_totrot,aobs = obs_qurotate(sobs_tot,yobs,keyvals)        
        
        ## placeholder for [update issuemask]


        if params.verbose >=1:
            if params.verbose >= 3:
                print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL OBS PREPROCESS INTEGRATION AND ROTATION')
            print('------------------------------------\n-----OBS PREPROCESS FINALIZED-------\n------------------------------------')

        return sobs_tot,yobs,rms,background,keyvals,sobs_totrot,aobs
    else:

        ## placeholder for [update issuemask]

        if params.verbose >=1:
            if params.verbose >= 3:
                print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL OBS PREPROCESS AND INTEGRATION')
            print('------------------------------------\n-----OBS PREPROCESS FINALIZED-------\n------------------------------------')

        return sobs_tot,yobs,rms,background,keyvals,np.zeros((sobs_tot.shape)),np.zeros((yobs.shape)) ## return the 0 arrays to keep returns consistent between 1 and 2 line inputs (it helps numba/jit).
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)    
def sdb_preprocess(yobs,keyvals,params):
## Main script to find and preload all necessary databases. 
## Returns databases as a list along with an encoding corresponding to each voxel of the observation
## this is made compatible to Numba object mode with loop-lifting to read all the necessary database in a parallel fashion

    if params.verbose >=1: 
        print('------------------------------------\n----------DB READ START-------------\n------------------------------------')
        if params.verbose >= 3:start0=time.time()        
    ######################################################################
    ## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx = keyvals[0]
    ny = keyvals[1]
    nline = keyvals[3]
    tline = keyvals[4]

    ######################################################################
    ## Preprocess the string information
    dbnames,dbynumbers,dbsubdirs=sdb_fileingest(params.dbdir,nline,tline,params.verbose)
    if params.verbose >=1:print("CLEDB covers a span of",dbynumbers.shape[0],"heights between",1+np.min(dbynumbers)/1000,"-",1+np.max(dbynumbers)/1000,"Solar radius")
   
    ######################################################################
    ## Create a file encoding showing which database to read for which voxel
    ## the encoding labels at this stage do not have an order.
    db_enc=np.zeros((nx,ny),dtype=np.int32)         ## location in outputted database list; OUTPUT VARIABLE
    db_enc_flnm=np.zeros((nx,ny),dtype=np.int32)    ## location in file list when read from disk; used only internally in sdb_preprocess
    
    for xx in range(nx):
        for yy in prange(ny):                     
            db_enc_flnm[xx,yy]=sdb_findelongation(yobs[xx,yy],dbynumbers)   

    db_uniq=np.unique(db_enc_flnm) ## makes a list of unique databases to read; the full list might have repeated entries
    if params.verbose >= 1: print("Load DB datafiles for",db_uniq.shape[0]," heights in memory for",nline,"line(s).\n------------------------------------")

    ######################################################################
    ## read the required databases and their header----------

    ## preprocess and return the database header information
    ## when multiple databases we assume all are of the same size.
    ##read the header and dimensions and just pass them as parameters to function calls
    if nline == 2:
        if dbsubdirs=="twolineDB":
            dbhdr=[sdb_parseheader(params.dbdir+'db.hdr')][0] ## dont understand why the 0 indexing is needed...    
            database0=[None]*db_uniq.shape[0]
            for ii in prange(db_uniq.shape[0]):
                database0[ii]=sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose)
                for ij in prange(7,-1,-1):
                    database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
        else:
            dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
            database0=[None]*db_uniq.shape[0]
            for ii in prange(db_uniq.shape[0]):
                database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=4) 
                for ij in prange(7,-1,-1):
                    database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
    elif nline == 1:
        dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
        database0=[None]*db_uniq.shape[0]
        for ii in prange(db_uniq.shape[0]):
            database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=4) 
            for ij in prange(7,-1,-1):
                database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]

    ## numpy large array implementation does not parallelize properly leading to a 5x increase in runtime per 1024 calculations
    ## reverted to use a list to feed the database set to the calculation

    ## Create a db_enc that is corresponding to the location in the outputted database list 
    for kk in prange(db_uniq.shape[0]):
        db_enc[np.where(db_enc_flnm == db_uniq[kk])] = kk
    
    ## standard reflected lists will be deprecated in numba 0.54; currently running 0.53. this is a fix!
    database = List()                            ## this is the List object implemented by numba
    [database.append(x) for x in database0]   

    ## [update issue mask] to implement


    if params.verbose >=1:
        if params.verbose >= 3:
            print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL DB SEARCH AND FIND')
        print('------------------------------------\n--------DB READ FINALIZED-----------\n------------------------------------')
    return db_enc,database,dbhdr
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
def sdb_fileingest(dbdir,nline,tline,verbose):  
#returns filename and index of elongation y in database  
## This prepares the database directory files outside of numba non-python..
    
##  Read the directory structure and see what lines are available.
    line_str=["fe-xiii_1074/","fe-xiii_1079/","mg-viii_3028/","si-ix_3934/","si-x_1430/"]
    line_bin=[]
    dbsubdirs=[]
    for i in line_str:
        line_bin.append(os.path.isdir(dbdir+i))

## two line db prepare
    if nline == 2:
        if sum(line_bin) >=2:
            for i in range(len(np.where(line_bin)[0])):
                if line_str[i][:-1] == tline[0] or line_str[i][:-1] == tline[1]:      ## the [:-1] indexing just removes the / from the filename to compare with the header keyword
                    dbsubdirs.append(line_str[np.where(line_bin)[0][i]])

            namesA=glob.glob(dbdir+dbsubdirs[0]+"DB*.DAT")
            namesB=glob.glob(dbdir+dbsubdirs[1]+"DB*.DAT")
            nn=str.find(namesA[0],'DB0')
            dbynumbers=np.empty(len(namesA),dtype=np.int32)
            for i in range(0,len(namesA)):
                dbynumbers[i]=np.int32(namesA[i][nn+2:nn+6])
            return [namesA,namesB],dbynumbers,dbsubdirs

        ## legacy loop to read two line databases (for fe XIII). .hdr and .DAT database files need to be in dbdir.
        elif os.path.isfile(dbdir+"db.hdr") and glob.glob(dbdir+"DB*.DAT") != []:
            names=glob.glob(dbdir+"DB*.DAT")
            nn=str.find(names[0],'DB0')
            dbynumbers=np.empty(len(names),dtype=np.int32)
            for i in range(0,len(names)):
                dbynumbers[i]=np.int32(names[i][nn+2:nn+6])
            return [names,None],dbynumbers,'twolineDB' ##double return of names +none is superfluous; reason is to keep returns consistent regardless in terms of datatype of the if case

        elif sum(line_bin) ==1:
            if verbose >=1: print("Two line observation provided! Requires two individual databases in directory")    
            return [None,None],None,'ingest error' 

        else: 
            if verbose >=1:print("No database or incomplete calculations found in directory ")
            return [None,None],None,'ingest error' 

## one line db prepare
    elif nline ==1:
        if sum(line_bin) >=1:
            for i in range(len(np.where(line_bin)[0])):
                if line_bin[i] == tline:
                    dbsubdirs.append(line_str[np.where(line_bin)[0][i]])

            namesA=glob.glob(dbdir+dbsubdirs+"DB*.DAT")
            nn=str.find(namesA[0],'DB0')
            dbynumbers=np.empty(len(namesA),dtype=np.int32)
            for i in range(0,len(namesA)):
                dbynumbers[i]=np.int32(namesA[i][nn+2:nn+6])
            return [namesA,None],dbynumbers,dbsubdirs

        else:
            if verbose >=1:print("No database or incomplete calculations found in directory ")
            return [None,None],None,'Ingest error' 
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)               ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_findelongation(y,dbynumbers):
# returns index of elongation y in database
# The calculation is faster ingested as a function rather than an inline calculation! 
    return np.argmin(np.abs(1. + dbynumbers / 1000. - y))
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=False,forceobj=True,cache=params.jitcache)  ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_parseheader(dbheaderfile):
## reads and parses the header and returns the parameters contained in the db.hdr of a database directory           
## db.hdr text file needs to have the specific version format described in the CLEDB 2.0.2 readme

    g=np.fromfile(dbheaderfile,dtype=np.float32,sep=' ')
## two kinds of data are returned
## 1. the linear coefficents of the form min, max, nx,theta,phi
## 2. logarithmically spaced parameters xed  (electron density array)
## for these also the min and max and number are also returned.
    ned=np.int32(g[0]) 
    ngx=np.int32(g[1])
    nbphi=np.int32(g[2])
    nbtheta=np.int32(g[3])
    emin=np.float32(g[4])
    emax=np.float32(g[5])
    xed=sdb_lcgrid(emin,emax,ned)    # Ne is log
    gxmin=np.float32(g[6])
    gxmax=np.float32(g[7])
    bphimin=np.float32(g[8])
    bphimax=np.float32(g[9])
    bthetamin=np.float32(g[10])
    bthetamax=np.float32(g[11])
    nline=np.int32(g[12])
    wavel=np.empty(nline,dtype=np.float32)
    
    for k in range(0,nline): wavel[k]=g[13+k]    

    return g, ned, ngx, nbphi, nbtheta, xed, gxmin,gxmax, bphimin, bphimax,\
        bthetamin, bthetamax, nline, wavel 
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,cache=params.jitcache)   # don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_read(fildb,dbhdr,verbose):
## here reading data is done and stored in variable database of size (ncalc,line,4)
## due to calling dcompress nad np.fromfile, this is incompatible with numba    
    if verbose >=3: start=time.time()    
    
    ######################################################################
    ## unpack dbcgrid parameters from the database accompanying db.hdr file 
    ##  
    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
    bthetamin, bthetamax, nline, wavel  = dbhdr 

    ## here reading data is done and stored in variable database of size (n,2*nline)

    if verbose >=1:
        print("INDIVIDUAL DB file location:", fildb)      
    ##CLE database compresion introduced numerical instabilities at small values. 
    ## It has been DISABLED in the CLE >=2.0.4 database building and CLEDB commit>=update-iqud
    #db=sdb_dcompress(np.fromfile(fildb, dtype=np.int16),verbose)
    db=np.fromfile(fildb, dtype=np.float32)
    
    if verbose >= 3: print("{:4.6f}".format(time.time()-start),' SECONDS FOR INDIVIDUAL DB READ\n----------')

    return np.reshape(db,(ned,ngx,nbphi,nbtheta,nline*4))
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)                              ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_lcgrid(mn,mx,n):
## grid for ned densities from mn to mx logarithms of density.

    ff=10.** ((mx-mn)/(n-1))
    xf=np.empty(n,np.float32)
    xf[0]=10.**mn
    for k in range(1,n): xf[k]=xf[k-1]*ff

    return xf
###########################################################################
###########################################################################

###########################################################################
###########################################################################
#@jit(parallel=False,forceobj=True,cache=params.jitcache) # don't try to parallelize things that don't need as the overhead will slow everything down
##The del command makes this incompatible with numba. The del command is needed.
def sdb_dcompress(i,verbose):
###~~~~~~~~~ DISABLED FOR CLE DATABASES NEWER THAN 2.0.4~~~~~~~~~~~~~~~~~~~~~
## helper routine for sdb_read
## CLEDB writes compressed databases to save storage space. We need this function to read the data
## This is numba incompatible due to the requirement to del negv

    cnst=-2.302585092994046*15./32767.
    ## Constants here must correspond to those in dbe.f in the CLE main directory
    ## c=-2.302585092994046 is the constant for the exponential; e.g. e^c=0.1
    ## 32767 is the limit of 4-byte ints
    ## 15 is the number of orders of magnitude for the range of intensities

    if verbose >=2:
        print(np.int64(sys.getsizeof(i)/1.e6)," MB in DB file")
        if np.int64(sys.getsizeof(i)/1.e6) > 250 : print("WARNING: Very large DB, Processing will be slow!")

    negv=np.flatnonzero(i < 0)

    f=np.abs(i)*cnst
    f=ne.evaluate("exp(f)")

    if size(negv) > 0:f[negv]=-f[negv]
    del negv

    return f
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache) ##Functiona like hasattr are not nopython compatible. Revert to object
def obs_headstructproc(sobs_in,headkeys,params):
## This is the ingestion for a minimal number of needed keywords from the observation headerfile
## Most keywords are defined statically as DKIST specs were not available at the time of this update. Dummy headkeys input should be used
## CoMP ingests most infoormation from its header

## unpacks the header metadata information from the observation
## Not fully implemented due to data and header structure not existing for DKIST.
    linestr = ["fe-xiii_1074","fe-xiii_1079","si-x_1430","si-ix_3934"]
    if len(sobs_in) == 2:
        nline = 2
        if params.integrated == True:                   ## for CoMP/uCoMP/COSMO
            if hasattr(headkeys[0],'wavetype') and hasattr(headkeys[1],'wavetype'):
                tline=[[i for i in linestr if str(headkeys[0].wavetype[0],'UTF-8') in i][0],[i for i in linestr if str(headkeys[1].wavetype[0],'UTF-8') in i][0]] 
                ## searches for the "wavetype" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed 
                ## the [0] unpacks the one element lists where used. 
            else:
                print("Can't read line information from keywords; aborting")
                return -1     #catastrophic exit
        else:
            tline=[linestr[0],linestr[1]]   ## for Cryo-NIRSP; static! no keywords yet implemented
    elif len(sobs_in) == 1:
        nline=1                                
        tline=["fe-xiii_1074"]   ## for Cryo-NIRSP; static! no keywords yet implemented
        ##Placeholder: one line setup for integrated or iuqd CoMP/uCoMP/COSMO data not yet implemented
        
    ## check if nline is correct. the downstream inversion modules are contingent on this keyword
    if nline != 1 and nline != 2:
        print("Not a one or two line observation; aborting")
        # placeholder for [update issuemask]
        return -1        #catastrophic exit

    if params.verbose >=1: 
        print('We are inverting observations of',nline,'coronal line(s) ')
        if nline >= 1:
            if   tline[0] == linestr[0]:
                print('Line 1: Fe XIII 1074.7nm')
            elif tline[0] == linestr[1]:
                print('Line 1: Fe XIII 1079.8nm')
            elif tline[0] == linestr[2]:
                print('Line 1: Si X    1430.1nm')
            elif tline[0] == linestr[3]:
                print('Line 1: Si IX   3934.3nm')
            if nline == 2:
                if   tline[1] == linestr[0]:
                    print('Line 2: Fe XIII 1074.7nm')
                elif tline[1] == linestr[1]:
                    print('Line 2: Fe XIII 1079.8nm')
                elif tline[1] == linestr[2]:
                    print('Line 2: Si X    1430.1nm')
                elif tline[1] == linestr[3]:
                    print('Line 2: Si IX   3934.3nm')

## array dimensions! 
    nx,ny=sobs_in[0].shape[0:2] 
    #nx,ny=sobs_in[0].shape[0:2] if sobs_in[0].shape[0] == headkeys[0].naxis1[0] and sobs_in[0].shape[1] == headkeys[0].naxis2[0] else print("Mismatch in array shapes; Fix before continuing") ## alternatively, these can be read from naxis[1-3]; Will be updfated when keywords are known for all sources
    if params.integrated != True:
        nw=sobs_in[0].shape[2]
        #nw=sobs_in[0].shape[2] if sobs_in[0].shape[2] == headkeys[0].naxis3[0] else print("Mismatch in array shapes; Fix before continuing")
    else:
        nw=1                        ## For line-integrated data; dont set 0 or arrays cant initialize

## array coordinate keywords
## These needs to be changed based on observation. For CLE this is normally the information in GRID.DAT
    if headkeys[0].instrume == "CLE-SIM" or headkeys[0].instrume  == "MUR-SIM":   #CASE 1: Simulated OBSERVATIONS  
        crpix1 = headkeys[0].crpix1                                                           ## Rerefence pixels along all dimensions
        crpix2 = headkeys[0].crpix2
        crpix3 = [np.int32(headkeys[0].crpix3),np.int32(headkeys[1].crpix3)]              ## two lines have different wavelength parameters


        crval1 = np.float32(headkeys[0].crval1)                                               ## solar/wavelength coordinates at crpixn in r_sun/angstrom; 
        crval2 = np.float32(headkeys[0].crval2)
        crval3 = [np.float32(headkeys[0].crval3), np.float32(headkeys[1].crval3)] 

        cdelt1 = np.float32(headkeys[0].cdelt1)                                               ## spatial/spectral resolution in R_sun/angstrom
        cdelt2 = np.float32(headkeys[0].cdelt2)
        cdelt3 = [np.float32(headkeys[0].cdelt3), np.float32(headkeys[1].cdelt3)]                                 

    elif (str(headkeys[0].instrume[0], "UTF-8") == "COMP"):                       #CASE 2: COMP OBSERVATIONS

        crpix1 = headkeys[0].crpix1[0]                                                        ## Comp takes reference at center
        crpix2 = headkeys[0].crpix2[0]
        crpix3 = [0,0]                                                                        ## not used here ## two lines have different wavelength parameters

        crval1 = np.float32(headkeys[0].crval1[0]*720/695700.0)                               ## solar coordinates at crpixn in r_sun; from -310.5*4.46 ##arcsec to R_sun conversion via 720/695700
        crval2 = np.float32(headkeys[0].crval2[0]*720/695700.0) 
        crval3 = [np.float32(headkeys[0].wave_ref[0]), np.float32(headkeys[1].wave_ref[0])]   ## not really used

        cdelt1 = np.float32(headkeys[0].cdelt1[0]*720/695700.0)                               ## COMP Cdelt in R_sun
        cdelt2 = np.float32(headkeys[0].cdelt2[0]*720/695700.0) 
        cdelt3 = [0.0,0.0]                                                                    ## not used for integrated data such as CoMP
    elif (str(headkeys[0].instrume[0], "UTF-8") == "Cryo-NIRSP"):                 #CASE 3: Cryo-NIRSP OBSERVATIONS
        ##To be updated
        print("Cryo-NIRSP header not yet implemented")
    else: print("Observation keywords not recognised.")                                       ## only CoMP, MURAM, and CLE examples are currently implemented

## Additional keywords of importance thatm might or might not be included in observations.
## check for the direction for the reference direction of linear polarization.
## angle direction is trigonometric; values are in radians
## 0 for horizontal ->0deg; np.pi/2 for vertical ->90deg rotation in linear polarization QU.
    linpolref = 0 #np.pi/2. <-- reference used in paraschiv & Judge 2022 for the direction used in computing the database (along Z axis)

## instrumental line broadening/width should be read and quantified here
## not clear at this point if this will be a constant or a varying keyword
    instwidth=0

    ## pack the decoded keywords into a comfortable python list variable to feed to downstream functions/modules
    return nx,ny,nw,nline,tline,crpix1,crpix2,crpix3,crval1,crval2,crval3,cdelt1,cdelt2,cdelt3,linpolref,instwidth
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_calcheight(keyvals):
## helper function for observation preprocessing.
## calculates the off-limb heights corresponding to each voxel.
## main assumption is that both observations have the same pointing.

    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny = keyvals[0:2]
    crpix1,crpix2=keyvals[5:7]
    crval1,crval2=keyvals[8:10]
    cdelt1,cdelt2=keyvals[11:13]

    if crpix1 != 0: 
        crval1 = crval1 - (crpix1 * cdelt1)
    if crpix2 != 0:
        crval2 = crval2 - (crpix2 * cdelt2)

    yobs=np.empty((nx,ny),dtype=np.float32)
    for xx in range(nx):
        for yy in prange(ny):
            yobs[xx,yy]=np.sqrt((crval1+(xx*cdelt1))**2+(crval2+(yy*cdelt2))**2 )

    return yobs
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_integrate(sobs_in,keyvals):
## reads the background and peak emission of lines, subtracts the background and integrates the signal.
## function was slow for big arrays. It has been rewritten for efficiency, but lost a lot in terms of clearness.
## Extended comments are provided for each step/component
## versions prior to "update-iqud" commit contain the original implementation.

## Sobs_in is an array of spatial size, wavelength size and 4 components, or a list of two such arrays for two-line obs.
## The lines are subscribed via ZZ. The last dimension is always 4 corresponding to Stokes IQUV 
## the output arrays are of last diimension 4 or 8 based on input. These are subscribed via iterating ZZ; 0:4 or 4:8 via (4*zz):(4*(zz+1))

######################################################################
## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny,nw = keyvals[0:3]
    nline = keyvals[3]

    sobs_tot=np.zeros((nx,ny,nline*4),dtype=np.float32)       ## output array of integrated profiles
    background=np.zeros((nx,ny,nline*4),dtype=np.float32)     ## output array to record the background levels in stokes I,Q,U,V, in that order
    rms=np.ones((nx,ny,nline*4),dtype=np.float32)             ## output array to record RMS statistics of the profile
    cdf=np.zeros((nx,ny,nw),dtype=np.float32)                 ## internal array to store CDF profiles
    #issuemask=np.zeros((4),dtype=np.int32)                   ## mask to test the statistical significance of the data: mask[0]=1 -> no signal above background in stokes I; mask[1]=1 -> distribution is not normal (skewed); mask[2]=1 line center (observed) not found; mask[3]=1=2 one or two fwhm edges were not found

######################################################################
## integrate for total profiles.
## Weirdly, with few operations per inner loop, pranges at all 3 for levels significantly reduce runtime to less than half!

    for zz in prange(nline): ## to travel through the two lines
        for xx in prange(nx): 
            for yy in prange(ny): 
                if ((np.count_nonzero(sobs_in[zz][xx,yy,:,0])) and (not(np.isnan(sobs_in[zz][xx,yy,:,:]).any()))):  ##enter only for pixels that have counts recorded
                    #compute the rough cdf distribution of the stokes I component to get a measure of the statistical noise.
                    cdf[xx,yy,:]=obs_cdf(sobs_in[zz][xx,yy,:,0]) ## keep CDF as a full array to 
 
                    ## compute the 1%-99% distributions to measure the quiet noise.
                    #l0=np.argwhere(np.abs(cdf[xx,yy,:]-0.01) == np.min(np.abs(cdf[xx,yy,:]-0.01)))[-1][0]
                    #r0=np.argwhere(np.abs(cdf[xx,yy,:]-0.99) == np.min(np.abs(cdf[xx,yy,:]-0.99)))[-1][0]
                    #l0=np.argwhere(np.abs(cdf[xx,yy,:]) >= 0.01)[0][0]  ## easier on the function calls than original version above
                    #r0=np.argwhere(np.abs(cdf[xx,yy,:]) >= 0.99)[0][0]
                    ## lr0 does a combined left-toright sort to not do repetitive argwhere calls. T
                    ##he two ends of lr0 give the start and end of above noise signal.
                    lr0=np.argwhere((np.abs(cdf[xx,yy,:]) > 0.01) & (np.abs(cdf[xx,yy,:]) <=0.99))
                        
                    ## remove the background noise from the profile. 
                    ## Take all the 4 positions denoting the IQUV measurements in one call.
                    ## sobs_in subscripted by zz,xx,yy, thus the sum is over columns(axis=0) over the wavelength range.
                    ## the sums give 4 values (for each iquv spectral signal) of noise before and after the line emission in the wavelength range. 
                    ## Finaly it divides by the amount of points summed over for each components.
                    background[xx,yy,(4*zz):(4*(zz+1))]=(np.sum(sobs_in[zz][xx,yy,0:lr0[0,0]+1,:],axis=0)+np.sum(sobs_in[zz][xx,yy,lr0[-1,0]:,:],axis=0))/(nw-lr0[-1,0]+lr0[0,0]+1)                  
                    ## Temporary array to store the background subtracted spectra. All four component done in one call.
                    tmp2=sobs_in[zz][xx,yy,:,:]-background[xx,yy,(4*zz):(4*(zz+1))]
                    
                    ##Now integrate/sum the four profiles
                    ## tmp2 [:,0] --> Stokes I 
                    sobs_tot[xx,yy,(4*zz)]=tmp2[tmp2[:,0]>0,0].sum()  ## background subtraction can introduce negative values unphysical for I; Sum only positive counts;

                    ## tmp2 [:,3] --> Stokes V
                    svmin=np.argwhere(tmp2[:,3] == np.min(tmp2[:,3]) )[-1,0]     ## position of minimum/negative Stokes V lobe
                    svmax=np.argwhere(tmp2[:,3] == np.max(tmp2[:,3]) )[-1,0]     ## position of maximum/positive Stokes V lobe
                    sobs_tot[xx,yy,(4*zz)+3]=(np.sign(svmin-svmax))*np.sum(np.fabs(tmp2[:,3]))   ## Sum the absolute of Stokes V signal and assign sign based on svmin and svmax lobe positions

                    ## tmp2 [:,1:3] --> Stokes Q and U
                    sobs_tot[xx,yy,(4*zz)+1:(4*zz)+3]=np.sum(tmp2[:,1:3],axis=0) ## sum both components in one call. These can take negative values, no need for precautions like for Stokes I.
                    
                    ## Compute the rms for each quantity. Ideally the background should be the same in all 4 measurements corresponding to one line.
                    ## Leave here commented lines for clarity of how RMS is computed.
                    ## First, here is variance of each state with detector/physical counts
                    ## Here is variance varis in ((I+S) - (I-S))/2 = S, in counts
                    #variance = np.abs((background[xx,yy,0] +background[xx,yy,0])/2.) ##
                    ## here is the variance in y = S/I:  var(y)/y^2 = var(S)/S^2 + var(I)/I^2
                    #var =  variance/sobs_tot[xx,yy,(4*zz):(4*(zz+1))]**2 + variance/sobs_tot[xx,yy,0]**2
                    #var *= (sobs_tot[xx,yy,(4*zz):(4*(zz+1))]/sobs_tot[xx,yy,0])**2 ## this is  form of normalization that is not required as the data is not normalized here.
                    #rms[xx,yy,(4*zz):(4*(zz+1))]=np.sqrt(var)
                    #rms[xx,yy,(4*zz):(4*(zz+1))]=np.sqrt((background[xx,yy,0]/sobs_tot[xx,yy,(4*zz):(4*(zz+1))]**2 + background[xx,yy,0]/sobs_tot[xx,yy,0]**2)*((sobs_tot[xx,yy,(4*zz):(4*(zz+1))]/sobs_tot[xx,yy,0])**2)) ## One line rms calculation; this is to save as much computation time possible
                    #rms[xx,yy,i+(4*zz)]=np.sqrt(((sobs_in[zz][xx,yy,0:l0+1,i]**2).mean()+(sobs_in[zz][xx,yy,r0:,i]**2).mean())/2.) ## canonical RMS estimation; not the same as implementation above
                # else:
                #     issuemask.....
    ## RMS is dependent on background and sobs_tot of first line ([xx,xy,0]) Because zz parralelization above might compute second line first, an error is introduced.
    ## This extra trtaversal removes the RMS calculation from the fast loops to avoid this issue. It only adds few seconds in computing time.
    for xx in prange(nx): 
        for yy in prange(ny): 
            rms[xx,yy,:]=np.sqrt((background[xx,yy,0]/sobs_tot[xx,yy,:]**2 + background[xx,yy,0]/sobs_tot[xx,yy,0]**2)*((sobs_tot[xx,yy,:]/sobs_tot[xx,yy,0])**2)) ## One line rms calculation            
    return sobs_tot,rms,background
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_qurotate(sobs_tot,yobs,keyvals):
## rotates the linear polarization components with the corresponding angle
## this implicitly assumes that yobs is kept in the same units as the header keys, either R_sun or arcsec.

    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny = keyvals[0:2]
    crpix1,crpix2=keyvals[5:7]
    crval1,crval2=keyvals[8:10]
    cdelt1,cdelt2=keyvals[11:13]
    linpolref=keyvals[14]

    if crpix1 != 0: 
        crval1 = crval1 - (crpix1 * cdelt1)
    if crpix2 != 0:
        crval2 = crval2 - (crpix2 * cdelt2)

    aobs=np.empty((nx,ny),dtype=np.float32)
    ## copy the sobs array 
    sobs_totrot=np.copy(sobs_tot)

    for xx in range(nx):
        for yy in prange(ny):
            #aobs[xx,yy]=2*np.pi-2*np.pi-linpolref + np.arccos((crval1 +xx*cdelt1)/yobs[xx,yy])
            aobs[xx,yy]= - np.arctan2((crval1 +xx*cdelt1),(crval2 +yy*cdelt2)) ## assuming the standard convention for arcsecond coordinates <--> trigonometric circle
            if (aobs[xx,yy] <0):
                aobs[xx,yy] =2*np.pi + aobs[xx,yy]
            ## if reference direction is horizontal/trigonometric, angle will >360; 
            ## reduce the angle to 1 trigonometric cirle unit for easier reading.
            if aobs[xx,yy] < linpolref:
                aobs[xx,yy] = 2*np.pi-aobs[xx,yy]
            else:
                aobs[xx,yy] = aobs[xx,yy]+linpolref

            ## update the rotated arrays using eq. 9 & 10 from Paraschiv & Judge 2022
            ## Use sobs_tot for filling up as to not update the final array in between operations
            ## The 0,3,4,7 subscripts remain the same as sobs_tot
            sobs_totrot[xx,yy,1]=sobs_tot[xx,yy,1]*np.cos(2*aobs[xx,yy])-sobs_tot[xx,yy,2]*np.sin(2*aobs[xx,yy])
            sobs_totrot[xx,yy,2]=sobs_tot[xx,yy,1]*np.sin(2*aobs[xx,yy])+sobs_tot[xx,yy,2]*np.cos(2*aobs[xx,yy])
            sobs_totrot[xx,yy,5]=sobs_tot[xx,yy,5]*np.cos(2*aobs[xx,yy])-sobs_tot[xx,yy,6]*np.sin(2*aobs[xx,yy])
            sobs_totrot[xx,yy,6]=sobs_tot[xx,yy,5]*np.sin(2*aobs[xx,yy])+sobs_tot[xx,yy,6]*np.cos(2*aobs[xx,yy])

    #if verbose >= 1: print("\n Observation Q and U intensities are rotated by",linpolref,"degrees around the unit circle.")
    return sobs_totrot,aobs
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_cdf(spectr):
## computes the cdf distribution for a spectra
    cdf=np.zeros((spectr.shape[0]),dtype=np.float32)
    for i in prange (0,spectr.shape[0]):
        cdf[i]=np.sum(spectr[0:i+1])        ## need to check if should start from 0 or not.
    cdf=cdf[:]/cdf[-1]                      ## norm the cdf to simplify interpretation
    return cdf
###########################################################################
###########################################################################
