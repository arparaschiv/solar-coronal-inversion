# -*- coding: utf-8 -*-

###############################################################################
### Parallel implementation of utilities for CLEDB_PREPINV                  ###
###############################################################################


#Contact: Alin Paraschiv, National Solar Observatory
#         arparaschiv@nso.edu
#         paraschiv.alinrazvan+cledb@gmail.com


### Tested Requirements  ################################################
### CLEDB database as configured in CLE V2.0.2; (db2021 executable) 
# python                    3.8.5 
# scipy                     1.6.2
# numpy                     1.20.1 
# numba                     0.53.1 
# llvmlite                  0.36.0 (as numba dependency probably does not need separate installing)

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
## Full Parallelization will generally not be possible while time functions are enabled.



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
@jit(parallel=params.jitparallel,forceobj=True,looplift=True)
def sobs_preprocess(sobs_in,params):
## Main script to read data and prepare it for analysis.

    if params.verbose >=1: 
        print('------------------------------------\n--------OBS PROCESSING START--------\n------------------------------------')
        if params.verbose >= 3:start0=time.time()        

## unpacks the header metadata information from the observation
## Not fully implemented due to data and header structure not existing.
    if len(sobs_in) ==2:
        nline=2                                 ## static! not implemented
        tline=["fe-xiii_1074","fe-xiii_1079"]   ## static! not implemented
    elif len(sobs_in) ==1:
        nline=1                                 ## static! not implemented
        tline=["fe-xiii_1074"               ]   ## static! not implemented
    ## check if nline is correct. the downstream inversion modules are contingent on this keyword
    if nline != 1 and nline != 2:
        print("Not a one or two line observation; aborting")
        # placeholder for [update issuemask]
        #catastrophic output
        return 

    if params.verbose >=1: 
        print('We are inverting observations of',nline,'coronal line(s) ')
        if nline >= 1:
            if   tline[0] == "fe-xiii_1074":
                print('Line 1: Fe XIII 1074.7nm')
            elif tline[0] == "fe-xiii_1079":
                print('Line 1: Fe XIII 1079.8nm')
            elif tline[0] == "si-ix_3934":
                print('Line 1: Si X    1430.1nm')
            elif tline[0] == "si-x_1430":
                print('Line 1: Si IX   3934.3nm')
            if nline == 2:
                if   tline[1] == "fe-xiii_1074":
                    print('Line 2: Fe XIII 1074.7nm')
                elif tline[1] == "fe-xiii_1079":
                    print('Line 2: Fe XIII 1079.8nm')
                elif tline[1] == "si-ix_3934":
                    print('Line 2: Si X    1430.1nm')
                elif tline[1] == "si-x_1430":
                    print('Line 2: Si IX   3934.3nm')

## check for the direction for the reference direction of linear polarization.
## angle direction is trigonometric; values are in radians
## 0 for horizontal ->0deg; np.pi/2 for vertical ->90deg rotation in linear polarization QU.
    linpolref=0#np.pi/2. 
## linpolref=np.pi/2 <-- reference used in paraschiv & Judge 2021 for the direction used in computing the database (along Z axis)

## Cryo-NIRSP instrumental width should be read and quantified here
## not clear at this point if this will be a constant or a varying keyword
    instwidth=0

## MAIN ASUSMPTION IS THAT THE KEYWORDS COME AS ATTACHED TO THE ARRAY (E.G. ARRAY_EXTENDED METHOD)
## array dimensions! 
    nx,ny,nw=sobs_in[0].shape[0:3]     ## alternatively, these can be read from naxis[1-3] keywords

## array coordinate keywords
## These needs to be changed based on observation. For CLE this is normally the information in GRID.DAT
## Values for the 3 dipole simulation
## these are in R_sun units. Production will most probably use arcsecond units. These should work without changes if things are kept consistent.

    ## CASE 1: 3 DIPOLE CLE SIMULATION DATA
#     crpix1 = 0                          ## assuming the reference pixel is in the left bottom of the array
#     crpix2 = 0
#     crpix3 = [np.int32(nw/2)-1,np.int32(nw/2)-1]                      ## two lines have different wavelength parameters

#     ##python stores arrays differently than normal, x dimension is second and y dimension is first. the reversal of /nx and /ny follows this issue.
#     crval1 = -0.75#0.8                  ## solar coordinates at crpixn in r_sun
#     crval2 = 0.8#-0.75
#     crval3 = [1074.62686-0.0124, 1079.78047-0.0124]     ## from CLE outfile; reference wavelengths are in vacuum
    
#     cdelt1 = (0.75-(-0.75))/nx         ## from grid.dat; remember the python axis flipping
#     cdelt2 = (1.5-0.8)/ny         
#     cdelt3 = [0.0247, 0.0249]
       
    ## CASE 2: MURAM INPUT DATA
    crpix1 = 0                          ## assuming the reference pixel is in the left bottom of the array
    crpix2 = 0
    crpix3 = [0,0]                      ## two lines have different wavelength parameters

    ##python stores arrays differently than normal, x dimension is second and y dimension is first. the reversal of /nx and /ny follows this issue.
    crval1 = -0.071                     ## solar coordinates at crpixn in r_sun; from muram xvec and y vec arrays
    crval2 = 0.989
    crval3 = [1074.257137, 1079.420513] ## from MURAM wvvec1 and wvvec2 0 positions

    cdelt1 = 0.0001379                  ## from muram xvec, yvec, wvvec1 and wvvce2 arrays
    cdelt2 = 0.0000689
    cdelt3 = [0.0071641,0.0071985]    
    
## pack the decoded keywords into a comfortable python list to feed to downstream functions/modules
    keyvals=(nx,ny,nw,nline,tline,crpix1,crpix2,crpix3,crval1,crval2,crval3,cdelt1,cdelt2,cdelt3,linpolref,instwidth)

##calculate an observation height and integrate the stokes profiles
    yobs=obs_calcheight(keyvals)
    sobs_tot,rms,background=obs_integrate(sobs_in,keyvals)


    if nline == 2:
        ## for two-line inversions we require a rotation of the linear polarization Q and U components
        sobs_totrot,aobs=obs_qurotate(sobs_tot,yobs,keyvals)

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

        return sobs_tot,yobs,rms,background,keyvals,np.zeros((sobs_tot.shape)),np.zeros((yobs.shape)) ## return the 0 arrays to keep returns consistent (it helps numba/jit).
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True)    
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
    db_enc=np.zeros((nx,ny),dtype=np.int32)
    
    for xx in range(nx):
        for yy in prange(ny):                     
            db_enc[xx,yy]=sdb_findelongation(yobs[xx,yy],dbynumbers)   
    
    db_uniq=np.unique(db_enc) ## makes a list of unique databases to read
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
                for ij in range(7,-1,-1):
                    database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]

        else:
            dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
            database0=[None]*db_uniq.shape[0]
            for ii in prange(db_uniq.shape[0]):
                database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=4) 
                for ij in range(7,-1,-1):
                    database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]

    elif nline == 1:
        dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
        database0=[None]*db_uniq.shape[0]
        for ii in prange(db_uniq.shape[0]):
            database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=4) 
            for ij in range(7,-1,-1):
                database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]

    ## numpy large array implementation does not parallelize properly leading to a 5x increase in runtime per 1024 calculations
    ## reverted to use a list to feed the database set to the calculation

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
@jit(parallel=params.jitparallel,forceobj=True,looplift=True)
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
@njit(parallel=False)               ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_findelongation(y,dbynumbers):
# returns index of elongation y in database
# The calculation is faster ingested as a function rather than an inline calculation! 
    return np.argmin(np.abs(1. + dbynumbers / 1000. - y))
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=False,forceobj=True)  ## don't try to parallelize things that don't need as the overhead will slow everything down
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
    emin=np.float64(g[4])
    emax=np.float64(g[5])
    xed=sdb_lcgrid(emin,emax,ned)    # Ne is log
    gxmin=np.float64(g[6])
    gxmax=np.float64(g[7])
    bphimin=np.float64(g[8])
    bphimax=np.float64(g[9])
    bthetamin=np.float64(g[10])
    bthetamax=np.float64(g[11])
    nline=np.int32(g[12])
    wavel=np.empty(nline,dtype=np.float64)
    
    for k in range(0,nline): wavel[k]=g[13+k]    

    return g, ned, ngx, nbphi, nbtheta, xed, gxmin,gxmax, bphimin, bphimax,\
        bthetamin, bthetamax, nline, wavel 
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=False,forceobj=True)   # don't try to parallelize things that don't need as the overhead will slow everything down
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
   
    db=sdb_dcompress(np.fromfile(fildb, dtype=np.int16),verbose)

    if verbose >= 3: print("{:4.6f}".format(time.time()-start),' SECONDS FOR INDIVIDUAL DB READ\n----------')

    #return np.reshape(db,(ned*ngx*nbphi*nbtheta,nline*4))
    return np.reshape(db,(ned,ngx,nbphi,nbtheta,nline*4))
###########################################################################
###########################################################################

###########################################################################
###########################################################################
#@jit(parallel=False,forceobj=True) # don't try to parallelize things that don't need as the overhead will slow everything down
##The del command makes this incompatible with numba. The del command is needed.
def sdb_dcompress(i,verbose):
## helper routine for sdb_read
## CLEDB writes compressed databases to save storage space. We need this function to read the data
## This is numba incompatible due to the requirement to del negv

    cnst=-2.302585092994046*15./32767.
    ## Constants here must correspond to those in dbe.f in the CLE main directory
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
@njit(parallel=False)                              ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_lcgrid(mn,mx,n):
## grid for ned densities from mn to mx logarithms of density.

    ff=10.** ((mx-mn)/(n-1))
    xf=np.empty(n,np.float64)
    xf[0]=10.**mn
    for k in range(1,n): xf[k]=xf[k-1]*ff

    return xf
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel)
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

    yobs=np.empty((nx,ny),dtype=np.float64)
    for xx in range(nx):
        for yy in prange(ny):
            yobs[xx,yy]=np.sqrt((crval1+(xx*cdelt1))**2+(crval2+(yy*cdelt2))**2 )

    return yobs
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel)
def obs_integrate(sobs_in,keyvals):
## reads the background and peak emission of lines, subtracts the background and integrates the signal

######################################################################
## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny,nw = keyvals[0:3]
    nline = keyvals[3]

    sobs_tot=np.zeros((nx,ny,nline*4),dtype=np.float64)
    cdf=np.empty((nx,ny,nw),dtype=np.float64)
    background=np.zeros((nx,ny,nline*4),dtype=np.float64)    ## array to record the background levels in stokes I,Q,U,V, in that order.
    mask=np.zeros((4),dtype=np.int32)                   ## mask to test the statistical significance of the data: mask[0]=1 -> no signal above background in stokes I; mask[1]=1 -> distribution is not normal (skewed); mask[2]=1 line center (observed) not found; mask[3]=1=2 one or two fwhm edges were not found
    rms=np.ones((nx,ny,nline*4),dtype=np.float64)

######################################################################
##integrate for total profiles.
    for zz in range(nline):
        for xx in range(nx):
            for yy in prange(ny):
                #start=time.time()
                if np.count_nonzero(sobs_in[zz][xx,yy,:,0]):
                #compute the rough cdf distribution of the stokes I component to get a measure of the statistical noise.
                    cdf[xx,yy,:]=obs_cdf(sobs_in[zz][xx,yy,:,0]) 
                    ## compute the 5%-95% distributions to measure quiet noise.
                    l0=np.where(np.abs(cdf[xx,yy,:]-0.01) == np.min(np.abs(cdf[xx,yy,:]-0.01)))[0][ 0]
                    r0=np.where(np.abs(cdf[xx,yy,:]-0.99) == np.min(np.abs(cdf[xx,yy,:]-0.99)))[0][-1]
                    #print(xx,yy,cdf.shape,cdf[-1],l0,r0)
                    #print("{:4.6f}".format(time.time()-start),' SUBSET: SECONDS FOR CDF')
                    ## remove the background noise from the profile. the 4 positions denote the IQUV measurements, in order.
                    for i in range(4):
                        ## remove the background noise from the profile. the 4 positions denote the IQUV measurements, in order.
                        background[xx,yy,i+(4*zz)]=(sobs_in[zz][xx,yy,0:l0+1,i].mean()+sobs_in[zz][xx,yy,r0:,i].mean())/2.
                        tmp=sobs_in[zz][xx,yy,:,i]-background[xx,yy,i+(4*zz)]
                        ##now fix the possible negative values resulted from the background averaging in case of stokes I
                        if i == 0:
                            locs=np.where(tmp < 0)
                            tmp[locs]=1e-7
                            sobs_tot[xx,yy,i+(4*zz)]=np.sum(tmp)
                        ## in case of stokes V, we integrate the absolute signal while keeping the convention (-1) for -+ ordered lobes an (1) for +- ordered lobes.
                        elif i == 3:
                            svmin=np.where(tmp == np.min(tmp) )[0][0]
                            svmax=np.where(tmp == np.max(tmp) )[0][0]
                            if svmin < svmax:
                                sobs_tot[xx,yy,i+(4*zz)]=(-1)*np.sum(np.fabs(tmp))
                            else:
                                sobs_tot[xx,yy,i+(4*zz)]=( 1)*np.sum(np.fabs(tmp))
                        else:
                            sobs_tot[xx,yy,i+(4*zz)]=np.sum(tmp)
                        #print("{:4.6f}".format(time.time()-start),' SUBSET: SECONDS FOR INTEGRATE')

                        ## Compute the rms for each quantity. Ideally the background should be the same in all 4 measurements corresponding to one line.
                        # First, here is variance of each state with detector/physical counts
                        # Here is variance varis in ((I+S) - (I-S))/2 = S, in counts
                        variance = np.abs((background[xx,yy,0] +background[xx,yy,0])/2.)
                        # here is the variance in y = S/I:  var(y)/y^2 = var(S)/S^2 + var(I)/I^2
                        var =  variance/(sobs_tot[xx,yy,i+(4*zz)]**2) + variance/(sobs_tot[xx,yy,0+(4*zz)]**2)
                        #var *= (sobs_tot[xx,yy,i+(4*zz)]/sobs_tot[xx,yy,0+(4*zz)])**2 ## this is  form of normalization that is not required as the data is not normalized here.
                        rms[xx,yy,i+(4*zz)]=np.sqrt(var)
                        #print("{:4.6f}".format(time.time()-start),' SUBSET: SECONDS FOR VARIANCE')

    return sobs_tot,rms,background
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel)
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

    aobs=np.empty((nx,ny),dtype=np.float64)
    ## copy the sobs array 
    sobs_totrot=np.copy(sobs_tot)

    for xx in range(nx):
        for yy in prange(ny):
            aobs[xx,yy]=2*np.pi-2*np.pi-linpolref + np.arcsin((crval1 +xx*cdelt1)/yobs[xx,yy]) ## assuming the standard convention for arcsecond coordinates <--> trigonometric circle
            ## if reference direction is horizontal/trigonometric, angle will >360; 
            ## reduce the angle to 1 trigonometric cirle unit for easier reading.
            if aobs[xx,yy] < linpolref:
                aobs[xx,yy] = 2*np.pi-aobs[xx,yy]
            else:
                aobs[xx,yy] = aobs[xx,yy]+linpolref

            ## update the rotated arrays using eq. 7 & 8 from paraschiv & judge 2021
            sobs_totrot[xx,yy,1]=sobs_totrot[xx,yy,1]*np.cos(2*aobs[xx,yy])-sobs_totrot[xx,yy,2]*np.sin(2*aobs[xx,yy])
            sobs_totrot[xx,yy,2]=sobs_totrot[xx,yy,1]*np.sin(2*aobs[xx,yy])-sobs_totrot[xx,yy,2]*np.cos(2*aobs[xx,yy])
            sobs_totrot[xx,yy,5]=sobs_totrot[xx,yy,5]*np.cos(2*aobs[xx,yy])-sobs_totrot[xx,yy,6]*np.sin(2*aobs[xx,yy])
            sobs_totrot[xx,yy,6]=sobs_totrot[xx,yy,5]*np.sin(2*aobs[xx,yy])-sobs_totrot[xx,yy,6]*np.cos(2*aobs[xx,yy])

    #if verbose >= 1: print("\n Observation Q and U intensities are rotated by",linpolref,"degrees around the unit circle.")
    return sobs_totrot,aobs
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel)
def obs_cdf(spectr):
## computes the cdf distribution for a spectra
    cdf=np.zeros((spectr.shape[0]),dtype=np.float64)
    for i in prange (0,spectr.shape[0]):
        cdf[i]=np.sum(spectr[0:i+1])        ## need to check if should start from 0 or not.
    cdf=cdf[:]/cdf[-1]                      ## norm the cdf to simplify interpretation
    return cdf
###########################################################################
###########################################################################
