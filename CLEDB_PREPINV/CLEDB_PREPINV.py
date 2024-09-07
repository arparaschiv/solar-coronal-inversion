# -*- coding: utf-8 -*-

###############################################################################
### Parallel implementation of utilities for CLEDB_PREPINV                  ###
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
from pylab import *
import scipy    
import time
import glob
import os
import sys
import multiprocessing
#import numexpr as ne ## Disabled as of update-iqud. No database compression going forward

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

    if params.verbose >= 1: 
        print('------------------------------------\n----SOBS_PREPROCESS - READ START----\n------------------------------------')
        if params.verbose >= 2:start0=time.time()        

## unpack the minimal number of header keywords
    keyvals = obs_headstructproc(sobs_in,headkeys,params)
    if(keyvals == -1):
        if params.verbose >=1: 
            print('SOBS_PREPROCESS: FATAL! OBS KEYWORD PROCESSING FAIL. Aborting!')
        return -1,0,0,0,0,0,0,0
    
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
    ## both variables are now initialized inside obs_qurotate
        #sobs_totrot = np.copy(sobs_tot)
        #aobs        = np.zeros((keyvals[0],keyvals[1]),dtype=np.float32)
        
        sobs_totrot,aobs = obs_qurotate(sobs_tot,keyvals)  
        
    ## to efficiently match pycelp databases, we require an estimate of the apparent density of the plasma.
    ## A standalonve version of this function is available at: https://github.com/arparaschiv/FeXIII-coronal-density
    ## This is available only for the Fe XIII pair (CLEDB always ratios 1074/1079) and requires a CHIANTI  look-up table that is linked in this repository.    
        if keyvals[4] == ["fe-xiii_1074","fe-xiii_1079"] or keyvals[4] == ["fe-xiii_1079","fe-xiii_1074"]:
            dobs = obs_dens(sobs_totrot,yobs,keyvals,params)       
        else:
            if params.verbose >=1: 
                print('SOBS_PREPROCESS: Can NOT properly account for abundance between ion species.\nIncompatible with the PyCELP implementation.\nOnly CLE databases can theoretically be used with extreme caution (not recommended).')
            #return -1,0,0,0,0,0,0,0

        ## placeholder for [update issuemask]


        if params.verbose >=1:
            if params.verbose >= 2:
                print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL OBS PREPROCESS INTEGRATION AND ROTATION')
            print('------------------------------------\n--SOBS_PREPROCESS - READ FINALIZED--\n------------------------------------')

        return sobs_tot,yobs,rms,background,keyvals,sobs_totrot,aobs,dobs

    else:

        ## placeholder for [update issuemask]

        if params.verbose >=1:
            if params.verbose >= 2:
                print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL OBS PREPROCESS AND INTEGRATION')
            print('------------------------------------\n--SOBS_PREPROCESS - READ FINALIZED--\n-----------------------------------')

        return sobs_tot,yobs,rms,background,keyvals,np.zeros((sobs_tot.shape)),np.zeros((yobs.shape)),np.zeros((yobs.shape)) ## return the 0 arrays to keep returns consistent between 1 and 2 line inputs (it helps numba/jit).
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)    
def sdb_preprocess(yobs,dobs,keyvals,params):
## Main script to find and preload all necessary databases. 
## Returns databases as a list along with an encoding corresponding to each voxel of the observation
## this is made compatible to Numba object mode with loop-lifting to read all the necessary database in a parallel fashion

    if params.verbose >=1: 
        print('------------------------------------\n----SDB_PREPROCESS - READ START-----\n------------------------------------')
        if params.verbose >= 2: start0=time.time()        
    ######################################################################
    ## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx = keyvals[0]
    ny = keyvals[1]
    nline = keyvals[3]
    tline = keyvals[4]

    ######################################################################
    ## Preprocess the string information
    dbnames,dbynumbers,dbsubdirs=sdb_fileingest(params.dbdir,nline,tline,params.verbose)
    
    ## No database could be read? NO RUN!!!
    if dbsubdirs == 'Ingest Error':
        if params.verbose >=1: print("SDB_PREPROCESS: FATAL! Could not read database")
        return None,None,None
      
    ######################################################################
    ## Create a file encoding showing which database to read for which voxel
    ## the encoding labels at this stage do not have an order.
    db_enc=np.zeros((nx,ny),dtype=np.int32)         ## location in outputted database list; OUTPUT VARIABLE
    db_enc_flnm=np.zeros((nx,ny),dtype=np.int32)    ## location in file list when read from disk; used only internally in sdb_preprocess
    
    for xx in range(nx):
        for yy in prange(ny):                     
            db_enc_flnm[xx,yy]=sdb_findelongationanddens(yobs[xx,yy],dobs[xx,yy],dbynumbers)   

    db_uniq=np.unique(db_enc_flnm) ## makes a list of unique databases to read; the full list might have repeated entries

    if params.verbose >= 1: 
        print("CLEDB databases cover a span of",len(np.unique(dbynumbers[:,1])),"solar heights between",dbynumbers[0,1]/100,"-",dbynumbers[-1,1]/100," radius")    
        print("Load ",db_uniq.shape[0]," heights x densities  DB datafiles in memory for each of ",nline,"line(s).\n------------------------------------")

    ######################################################################
    ## read the required databases and their header----------

    ## preprocess and return the database header information
    ## when ingesting multiple databases we assume all are of the same size.
    ##read the header and dimensions and just pass them as parameters to function calls
    if nline == 2:
        if dbsubdirs=="twolineDB":
            dbhdr=[sdb_parseheader(params.dbdir+'db.hdr')][0] ## dont understand why the 0 indexing is needed...    
            database0=[None]*db_uniq.shape[0]
            for ii in prange(db_uniq.shape[0]):
                database0[ii]=sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose)
                for ij in prange(7,-1,-1):
                    if dbhdr[-1] == 0:
                        database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
                    elif dbhdr[-1] == 1:
                        database0[ii][:,:,:,ij]=database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
        else:
            dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
            database0=[None]*db_uniq.shape[0]
            for ii in prange(db_uniq.shape[0]):
                database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=-1) 
                for ij in prange(7,-1,-1):
                    if dbhdr[-1] == 0:
                        database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
                    elif dbhdr[-1] == 1:
                        database0[ii][:,:,:,ij]=database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
    elif nline == 1:
        dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
        database0=[None]*db_uniq.shape[0]
        for ii in prange(db_uniq.shape[0]):
            database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=-1) 
            for ij in prange(7,-1,-1):
                if dbhdr[-1] == 0:
                    database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
                elif dbhdr[-1] == 1:
                    database0[ii][:,:,:,ij]=database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]

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
        if params.verbose >= 2:
            print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL DB SEARCH AND FIND')
        print('------------------------------------\n--SDB_PREPROCESS - READ FINALIZED---\n------------------------------------')
    return db_enc,dbnames,db_enc_flnm,db_uniq,database,dbhdr
###########################################################################
###########################################################################

###########################################################################
###########################################################################
### Helper functions ######################################################
###########################################################################
###########################################################################

###########################################################################
###########################################################################
#@jit(parallel=params.jitparallel,forceobj=True,cache=params.jitcache)
def sdb_fileingest(dbdir,nline,tline,verbose):  
#returns filename and index of elongation y in database  
## This prepares the database directory files outside of numba non-python..
    
##  Read the directory structure and see what lines are available.
    linestr=["fe-xiii_1074/", "fe-xiii_1079/", "si-x_1430/", "si-ix_3934/", "mg-viii_3028/"]
    line_bin=[]
    dbsubdirs=[]
    for i in linestr:
        line_bin.append(os.path.isdir(dbdir+i))
    
## two line db prepare
    if nline == 2:
        if np.sum(line_bin) >= 2:
            for i in range(np.sum(line_bin)):
                if linestr[i][:-1] == tline[0] or linestr[i][:-1] == tline[1]:      ## the [:-1] indexing just removes the / from the filename to compare with the header keyword
                    dbsubdirs.append(linestr[np.where(line_bin)[0][i]])

            namesA     = sorted(glob.glob(dbdir+dbsubdirs[0]+"DB_*"))
            namesB     = sorted(glob.glob(dbdir+dbsubdirs[1]+"DB_*"))
            nn         = str.find(namesA[0],'DB_h')
            dbynumbers = np.empty((len(namesA),3),dtype=np.float32)
            
            for i in range(0,len(namesA)):
                dbynumbers[i,0] = i                                   ## database file index
                dbynumbers[i,1] = np.float32(namesA[i][nn+4:nn+7])    ## database file projected height at index
                dbynumbers[i,2] = np.float32(namesA[i][nn+9:-4])      ## database file density at index
                
            return [namesA,namesB],dbynumbers,dbsubdirs

        ## legacy loop to read two line databases (for Fe XIII). .hdr and .DAT database files need to be in dbdir without a specific ion subfolder.
        ## Disabled as this is not an offered database building option.
        # elif os.path.isfile(dbdir+"db.hdr") and glob.glob(dbdir+"DB*.DAT") != []:
        #     names=glob.glob(dbdir+"DB*.DAT")
        #     nn=str.find(names[0],'DB0')
        #     dbynumbers=np.empty(len(names),dtype=np.int32)
        #     for i in range(0,len(names)):
        #         dbynumbers[i]=np.int32(names[i][nn+2:nn+6])
        #     return [names,None],dbynumbers,'twolineDB' ##double return of names +none is superfluous; reason is to keep returns consistent regardless in terms of datatype of the if case

        elif sum(line_bin) ==1:
            if verbose >=2: print("SDB_FILEINGEST: FATAL! Two line observation provided! Requires two individual ion databases in directory. Only one database is computed.")
            return [None,None],None,'Ingest Error'

        else: 
            if verbose >=2: print("SDB_FILEINGEST: FATAL! No database or incomplete calculations found in directory ")
            return [None,None],None,'Ingest Error' 

## one line db prepare????
    ##  is there a need for a database solution for just one line?
    elif nline ==1:
        if sum(line_bin) >=1:
            for i in range(len(np.where(line_bin)[0])):
                if line_bin[i] == tline:
                    dbsubdirs.append(linestr[np.where(line_bin)[0][i]])

            namesA=glob.glob(dbdir+dbsubdirs+"DB*.DAT")
            nn=str.find(namesA[0],'DB0')
            dbynumbers=np.empty(len(namesA),dtype=np.int32)
            for i in range(0,len(namesA)):
                dbynumbers[i]=np.int32(namesA[i][nn+2:nn+6])
            return [namesA,None],dbynumbers,dbsubdirs

        else:
            if verbose >=2: print("SDB_FILEINGEST: FATAL! No database or incomplete calculations found in directory ")
            return [None,None],None,'Ingest Error' 
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)               ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_findelongationanddens(y,d,dbynumbers):
# returns index of elongation y and density d in database set
# The calculation is faster ingested as a function rather than an inline calculation! 
    yy = np.argwhere( np.abs( dbynumbers[:,1] / 100. - y ) == np.min(np.abs( dbynumbers[:,1] / 100. - y )) )  ## preselec for the observed height
    dd = np.argmin( np.abs( dbynumbers[yy[0][0]:yy[-1][0],2] / 100 - d ))                                     ## Out of yy, select the closest matching density
    ii = yy[0][0]+dd                                                                                          ## index is the sum of the starting height index and the density match.
    return np.int32(dbynumbers[ii,0])
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
## 2. logarithmically spaced parameters xed  (electron density array in CLE case); for these the min and max and number are not returned.

    ngy       = np.int32(g[0])             ## database is individual with respect with ngy, no need to output
    ned       = np.int32(g[1])             ## pycelp databases are individual also with respect to density. output only for consistency with cle
    ngx       = np.int32(g[2])
    nbphi     = np.int32(g[3])
    nbtheta   = np.int32(g[4])
    emin      = np.float32(g[5])
    emax      = np.float32(g[6])
    xed       = sdb_lcgrid(emin,emax,ned)    ## Ne is log
    gxmin     = np.float32(g[7])
    gxmax     = np.float32(g[8])
    bphimin   = np.float32(g[9])
    bphimax   = np.float32(g[10])
    bthetamin = np.float32(g[11])
    bthetamax = np.float32(g[12])
    nline     = np.int32(g[13])
    wavel     = np.empty(nline,dtype=np.float32)
    dbtype    = np.int32(g[-1])    # if g[-1] == 0: --> CLE database##   elif g[-1] == 1: --> PyCELP database
    
    for k in range(0,nline): wavel[k]=g[14+k]

    return g, ned, ngx, nbphi, nbtheta, xed, gxmin, gxmax, bphimin, bphimax,\
        bthetamin, bthetamax, nline, wavel,dbtype
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,cache=params.jitcache)   # don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_read(fildb,dbhdr,verbose):
## here reading data is done and stored in variable database of size (ncalc,line,4)
## due to calling dcompress nad np.fromfile, this is incompatible with numba    
    if verbose >=2: start=time.time()    
    
    ######################################################################
    ## unpack dbcgrid parameters from the database accompanying db.hdr file 
    ##  
    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
    bthetamin, bthetamax, nline, wavel,dbtype  = dbhdr 

    ## here reading data is done and stored in variable database of size (n,2*nline)

    if verbose >=1:
        print("INDIVIDUAL DB file location:", fildb)      
    ##CLE database compresion introduced numerical instabilities at small values. 
    ## It has been DISABLED in the CLE >=2.0.4 database building and CLEDB commit>=update-iqud
    #db=sdb_dcompress(np.fromfile(fildb, dtype=np.int16),verbose)
    
    if dbtype == 0:
        db = np.fromfile(fildb, dtype=np.float32)
    elif dbtype == 1:
        db = np.load(fildb) 

    if verbose >=2:
        print("SDB_READ: ",np.int64(sys.getsizeof(db)/1.e6)," MB in DB file")
        if np.int64(sys.getsizeof(db)/1.e6) > 250 : print("SDB_READ: WARNING! Very large DB. Lots of RAM are required. Processing will be slow!")

    
    if verbose >= 2: print("{:4.6f}".format(time.time()-start),' SECONDS FOR INDIVIDUAL DB READ\n----------')

    if dbtype == 0:
        return np.reshape(db,(ned,ngx,nbphi,nbtheta,nline*4))
    elif dbtype == 1:
        return db
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
# def sdb_dcompress(i,verbose):
# ###~~~~~~~~~ DISABLED FOR CLE DATABASES NEWER THAN 2.0.4~~~~~~~~~~~~~~~~~~~~~
# ## helper routine for sdb_read
# ## CLEDB writes compressed databases to save storage space. We need this function to read the data
# ## This is numba incompatible due to the requirement to del negv

#     cnst=-2.302585092994046*15./32767.
#     ## Constants here must correspond to those in dbe.f in the CLE main directory
#     ## c=-2.302585092994046 is the constant for the exponential; e.g. e^c=0.1
#     ## 32767 is the limit of 4-byte ints
#     ## 15 is the number of orders of magnitude for the range of intensities

#     if verbose >=2:
#         print(np.int64(sys.getsizeof(i)/1.e6)," MB in DB file")
#         if np.int64(sys.getsizeof(i)/1.e6) > 250 : print("WARNING: Very large DB, Processing will be slow!")

#     negv=np.flatnonzero(i < 0)

#     f=np.abs(i)*cnst
#     f=ne.evaluate("exp(f)")

#     if size(negv) > 0:f[negv]=-f[negv]
#     del negv

#     return np.float32(f)   ## the code expects real variables moving forward.
###########################################################################
###########################################################################

###########################################################################
###########################################################################
##@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache) ##Functiona like hasattr are not non-python compatible. Revert to object
## can not be numba object due to CoMP recarrays containing dtypes
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
            if hasattr(headkeys[0],'WAVETYPE') and hasattr(headkeys[1],'WAVETYPE'):
                tline=[[i for i in linestr if str(headkeys[0].wavetype[0],'UTF-8') in i][0],[i for i in linestr if str(headkeys[1].wavetype[0],'UTF-8') in i][0]] 
                ## searches for the "wavetype" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed 
                ## the [0] unpacks the one element lists where used. 
            elif hasattr(headkeys[0],'FILTER') and hasattr(headkeys[1],'FILTER'):
                tline=[[i for i in linestr if str(headkeys[0].filter[0],'UTF-8') in i][0],[i for i in linestr if str(headkeys[1].filter[0],'UTF-8') in i][0]] 
                ## searches for the "FILTER" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed 
                ## the [0] unpacks the one element lists where used. 
            else:
                if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: FATAL! Can't read line information from keywords.")
                return -1     #catastrophic exit
        else:
            tline=[linestr[0],linestr[1]]   ## for Cryo-NIRSP; static! no keywords yet implemented
    elif len(sobs_in) == 1:
        nline=1                                
        tline=["fe-xiii_1074"]   ## for Cryo-NIRSP; static! no keywords yet implemented
        ##Placeholder: one line setup for integrated or iuqd CoMP/uCoMP/COSMO data not yet implemented

    ## check if nline is correct. the downstream inversion modules are contingent on this keyword
    if nline != 1 and nline != 2:
        if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: FATAL! Not a one or two line observation; aborting")
        # placeholder for [update issuemask]
        return -1        #catastrophic exit

    if params.verbose >= 1: 
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
    
    elif (str(headkeys[0].instrume[0], "UTF-8") == "UCoMP"):                       #CASE 3: uCoMP OBSERVATIONS;; not definitive. based on Steve's processing output keywords

        crpix1 = nx/2                                                                         ## Comp takes reference at center
        crpix2 = ny/2
        crpix3 = [0,0]                                                                        ## not used here ## two lines have different wavelength parameters

        crval1 = 0                                                                            ## solar coordinates at crpixn in r_sun; from -310.5*4.46 ##arcsec to R_sun conversion via 720/695700
        crval2 = 0
        crval3 = [np.float32(headkeys[0].filter[0]), np.float32(headkeys[1].filter[0])]   ## not really used

        cdelt1 = np.float32(headkeys[0].cdelt1[0]*720/695700.0)                               ## COMP Cdelt in R_sun
        cdelt2 = np.float32(headkeys[0].cdelt1[0]*720/695700.0) 
        cdelt3 = [0.0,0.0]     
    elif (str(headkeys[0].instrume[0], "UTF-8") == "Cryo-NIRSP"):                 #CASE 3: Cryo-NIRSP OBSERVATIONS
        ##To be updated
        if params.verbose >= 1: print("OBS_HEADSTRUCTPROC: FATAL! Cryo-NIRSP header not yet implemented")
    else: 
        if params.verbose >= 1: print("OBS_HEADSTRUCTPROC: FATAL! Observation keywords not recognised.") ## only CoMP, MURAM, and CLE examples are currently implemented

## Additional keywords of importance that might or might not be included in observations.
## assign the database reference direction of linear polarization. See CLE routine db.f line 120.
## Angle direction is trigonometric; values are in radians
## 0 for horizontal ->0deg; np.pi/2 for vertical ->90deg rotation in linear polarization QU.
    linpolref = params.dblinpolref ##  0 is the reference used in paraschiv & Judge 2022 for the direction used in computing the database (at Z=0 plane). 

## instrumental line broadening/width should be read and quantified here
## not clear at this point if this will be a constant or a varying keyword
## a 0 value will skip including asn instrumental contribution to computing non-thermal widths
    instwidth = params.instwidth 

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
    #cdf=np.zeros((nx,ny,nw),dtype=np.float32)                 ## internal array to store CDF profiles
    #issuemask=np.zeros((4),dtype=np.int32)                   ## mask to test the statistical significance of the data: mask[0]=1 -> no signal above background in stokes I; mask[1]=1 -> distribution is not normal (skewed); mask[2]=1 line center (observed) not found; mask[3]=1=2 one or two fwhm edges were not found

######################################################################
## integrate for total profiles.
## Weirdly, with few operations per inner loop, pranges at all 3 for levels significantly reduce runtime to less than half!

## set up the cpu worker and argument arrays
    p         = multiprocessing.Pool(processes=multiprocessing.cpu_count()-2,maxtasksperchild = 10000)     ## dynamically defined from system query as total CPU core number - 2
    ## argument index keeper for splitting tasks to cpu cores
    arg_array = []
    
    for zz in range(nline): ## to travel through the two lines
        for xx in range(nx): 
            for yy in range(ny): 
                arg_array.append((xx,yy,zz,nw,sobs_in[zz][xx,yy,:,:]))

    rs        = p.starmap(obs_integrate_work_1pix,arg_array)
    p.close()

    for i,res in enumerate(rs):
        xx,yy,zz,sobs_tot[xx,yy,(4*zz):(4*(zz+1))],rms[xx,yy,(4*zz):(4*(zz+1))],background[xx,yy,(4*zz):(4*(zz+1))] = res 

    return sobs_tot,rms,background
###########################################################################
###########################################################################

@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_integrate_work_1pix(xx,yy,zz,nw,sobs_in_1pix):
    if ((np.count_nonzero(sobs_in_1pix[:,0])) and (np.isfinite(sobs_in_1pix[:,:]).all())):  ## enter only for pixels that have counts recorded
        #compute the rough cdf distribution of the stokes I component to get a measure of the statistical noise.
        cdf = obs_cdf(sobs_in_1pix[:,0]) ## keep CDF as a full array to 
    
        ## compute the 1%-99% distributions to measure the quiet noise.
        #l0=np.argwhere(np.abs(cdf[xx,yy,:]-0.01) == np.min(np.abs(cdf[xx,yy,:]-0.01)))[-1][0]
        #r0=np.argwhere(np.abs(cdf[xx,yy,:]-0.99) == np.min(np.abs(cdf[xx,yy,:]-0.99)))[-1][0]
        #l0=np.argwhere(np.abs(cdf[xx,yy,:]) >= 0.01)[0][0]  ## easier on the function calls than original version above
        #r0=np.argwhere(np.abs(cdf[xx,yy,:]) >= 0.99)[0][0]
        ## lr0 does a combined left-toright sort to not do repetitive argwhere calls. T
        ##he two ends of lr0 give the start and end of above noise signal.
        lr0 = np.argwhere((np.abs(cdf) > 0.01) & (np.abs(cdf) <=0.99))
            
        ## remove the background noise from the profile. 
        ## Take all the 4 positions denoting the IQUV measurements in one call.
        ## sobs_in subscripted by zz,xx,yy, thus the sum is over columns(axis=0) over the wavelength range.
        ## the sums give 4 values (for each iquv spectral signal) of noise before and after the line emission in the wavelength range. 
        ## Finaly it divides by the amount of points summed over for each components.
        background_1pix = (np.sum(sobs_in_1pix[0:lr0[0,0]+1,:],axis=0)+np.sum(sobs_in_1pix[lr0[-1,0]:,:],axis=0))/(nw-lr0[-1,0]+lr0[0,0]+1)                  
        ## Temporary array to store the background subtracted spectra. All four component done in one call.
        tmp2=sobs_in_1pix[:,:]-background_1pix
        
        ##Now integrate/sum the four profiles
        ## tmp2 [:,0] --> Stokes I 
        sobs_tot_1pix=np.zeros((4),dtype=np.float32)
        sobs_tot_1pix[0]=np.sum(tmp2[tmp2[:,0]>0,0])           ## background subtraction can introduce negative values unphysical for I; Sum only positive counts;
    
        ## tmp2 [:,3] --> Stokes V
        svmin=np.argwhere(tmp2[:,3] == np.min(tmp2[:,3]) )[-1,0]     ## position of minimum/negative Stokes V lobe
        svmax=np.argwhere(tmp2[:,3] == np.max(tmp2[:,3]) )[-1,0]     ## position of maximum/positive Stokes V lobe
        sobs_tot_1pix[3]=(np.sign(svmin-svmax))*np.sum(np.fabs(tmp2[:,3]))   ## Sum the absolute of Stokes V signal and assign sign based on svmin and svmax lobe positions
    
        ## tmp2 [:,1:3] --> Stokes Q and U
        sobs_tot_1pix[1:3]=np.sum(tmp2[:,1:3],axis=0) ## sum both components in one call. These can take negative values, no need for precautions like for Stokes I.
        
        ## Compute the rms for each quantity. Ideally the background should be the same in all 4 measurements corresponding to one line.
        ## Leave here commented lines for clarity of how RMS is computed.
        ## First, here is variance of each state with detector/physical counts
        ## Here is variance varis in ((I+S) - (I-S))/2 = S, in counts
        # variance = np.abs((background[xx,yy,0] +background[xx,yy,0])/2.) ##
        # # here is the variance in y = S/I:  var(y)/y^2 = var(S)/S^2 + var(I)/I^2
        # var =  variance/sobs_tot[xx,yy,(4*zz):(4*(zz+1))]**2 + variance/sobs_tot[xx,yy,0]**2
        # var *= (sobs_tot[xx,yy,(4*zz):(4*(zz+1))]/sobs_tot[xx,yy,0])**2 
        # rms[xx,yy,(4*zz):(4*(zz+1))]=np.sqrt(var)
        #rms[xx,yy,(4*zz):(4*(zz+1))]=np.sqrt((background[xx,yy,0]/sobs_tot[xx,yy,(4*zz):(4*(zz+1))]**2 + background[xx,yy,0]/sobs_tot[xx,yy,0]**2)*((sobs_tot[xx,yy,(4*zz):(4*(zz+1))]/sobs_tot[xx,yy,0])**2)) ## One line rms calculation; this is to save as much computation time possible. Due to parallelization, this calculation needs to be done separately. see below
        #rms[xx,yy,i+(4*zz)]=np.sqrt(((sobs_in[zz][xx,yy,0:l0+1,i]**2).mean()+(sobs_in[zz][xx,yy,r0:,i]**2).mean())/2.) ## canonical RMS estimation; not the same as implementation above
        
        ## distribution standard deviation 
        # for kk in range(4):
        #     rms[xx,yy,(4*zz)+kk] = np.std(sobs_in[zz][xx,yy,:,kk]/np.max(np.abs(sobs_in[zz][xx,yy,:,kk]))) ##a standard deviation in normalized units as normalized data is matched in cledb_matchiquv
            #rms[xx,yy,(4*zz)+kk] = (0.5*np.std(sobs_in[zz][xx,yy,0:lr0[0,0]+1,kk]/np.max(np.abs(sobs_in[zz][xx,yy,0:lr0[0,0]+1,kk])))+0.5*np.std(sobs_in[zz][xx,yy,lr0[-1,0]:,kk]/np.max(np.abs(sobs_in[zz][xx,yy,lr0[-1,0]:,kk]))) )
    # else:
    #     issuemask.....

        rms_1pix = np.sqrt((background_1pix[0]/sobs_tot_1pix[:]**2 + background_1pix[0]/sobs_tot_1pix[0]**2)*((sobs_tot_1pix[:]/sobs_tot_1pix[0])**2)) ## One line rms calculation based on the above formulation    
    else:
        return xx,yy,zz,np.zeros((4),dtype=np.float32),np.zeros((4),dtype=np.float32),np.zeros((4),dtype=np.float32)

    return xx,yy,zz,sobs_tot_1pix,rms_1pix,background_1pix
###########################################################################
###########################################################################


###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_qurotate(sobs_tot,keyvals):
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
    sobs_totrot=np.copy(sobs_tot)             ## copy the sobs array as only QU need to be updated

    for xx in range(nx):
        for yy in prange(ny):
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
            
            ## It would be more efficient to normalize sobs_totrot here, but then the normalization factor can not be easily transported to cledb_quderotate to bring back the profiles that can be compared to sobs_tot.
            ## The normalization is done inside cledb_matchiquv or cledb_matchiqud
            ## sobs_totrot[xx,yy,:]=sobs_totrot[xx,yy,:]/sobs_totrot[xx,yy,np.argwhere(sobs_totrot[xx,yy,:] == np.max(sobs_totrot[xx,yy,:]))[0,0]]

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

###########################################################################
###########################################################################
def obs_dens(sobs_totrot,yobs,keyvals,params):
## compute the density from two stokes I observations (serialized parallel runs for 1 pixel at a time) of Fe XIII assuming:

    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny = keyvals[0:2]

    dobs=np.empty((nx,ny),dtype=np.float32)

    ## theoretical chianti ratio calculations created via PyCELP or SSWIDL        
    ## read the chianti table         
    if (params.lookuptb[-4:] == ".npz"):                         ## default
        chianti_table        =   dict(np.load(params.lookuptb))  ## variables (h,den,rat) directly readable by work_1pix .files required for loading the data directly.
    else:
        if params.verbose >= 1: print("OBS_DENS: FATAL! CHIANTI look-up table not found. Is the path correct?")
        return density                                ## a zero array at this point  
    #print(chianti_table.keys())                      ## Debug - check the arrays
    
    ##  in the look-up table:
    ## 'h' is an array of heights in solar radii ranged 1.01 - 2.00(pycelp)  
    ## 'den' is an array of density (ranged 6.00 to 12.00 (pycelp) or 5.0 to 13.0 (sswidl); the broader interval is not really recommended here. 
    ## 'rat' is a 2D array containing line ratios corresponding to the density range values at each distinct height.
    
    ##  To query the look-up table:
    ##  print(chia['h'],chia['h'].shape,chia['rat'].shape,chia['den'].shape)

    ## look-up table resolutions
    ## pycelp: h is of shape [99]; den and rat are arrays of shape [99, 120] corresponding to the 99 h height values and 120 density values

    ## set up the cpu worker and argument arrays
    p         = multiprocessing.Pool(processes=multiprocessing.cpu_count()-2,maxtasksperchild = 10000)     ## dynamically defined from system query as total CPU core number - 2
    ## argument index keeper for splitting tasks to cpu cores
    ## two branches to separate slingle slits vs rasters
    arg_array = []
                                                                                              ## Raster slit branch
    for xx in range(sobs_totrot.shape[0]): 
        for yy in range(sobs_totrot.shape[1]): 
            arg_array.append((xx,yy,sobs_totrot[xx,yy,0],sobs_totrot[xx,yy,4],chianti_table,yobs[xx,yy])) # 1074/1079 ratio

    rs        = p.starmap(obs_dens_work_1pix,arg_array)
    p.close()

    for i,res in enumerate(rs):
        xx,yy,dobs[xx,yy] = res 

    return np.round(np.log10(dobs),3)
###########################################################################
###########################################################################

###########################################################################
###########################################################################
def obs_dens_work_1pix(xx,yy,a_obs,b_obs,chianti_table,y_obs):
    ## compute the ratio of the observation ## 1074/1079 fraction, same as rat component of the chianti_table 
    
    ## Sanity checks at pixel level
    if b_obs == 0:                                      ## don't divide by 0
        return xx,yy,1                                  ## return 1 value <--np.log10(1) will give 0 in obs_dens


    ## line ratio calculations for peak and integrated quantities
    ## line ratio calculations of integrated line counts, requires amplitude, distribution center and sigma parameters.
    ## There are two ways of doing this; setting a wavelength linearspace and then sum the gaussian over that inverval, OR take the analytical gaussian integral (seems faster).
    ## to not have to load header extensions, we hardcode the central wavelngth sampling to 1074.7 and 1079.8 respectively.
    ## Distribution center and sigma need to be changed from km/s units to nm units.

    ## line ratio calculations for peak quantities only 
    rat_obs       = a_obs/b_obs        ## requires only one input number for each line 
    #rat_obs_noise = rat_obs*np.sqrt((np.sqrt(a_obs)/a_obs)**2+(np.sqrt(b_obs)/b_obs)**2)    ## error propagation
    
    ## another sanity check for numerical issues
    if (np.isnan(rat_obs) or np.isinf(rat_obs)):        ## discard nans and infs ratios
        return xx,yy,0                                  ## return 0 value
    
    else:                                               ## main loop for valid "rat_obs" value
        ## find the corresponding height (in solar radii) for each pixel
        subh = np.argwhere(chianti_table['h'] > y_obs)                                                                                 
        
        ## if height is greater than maximum h (2.0R_sun as in the currently implemented table) just use the 2.0R_sun corresponding ratios.
        if len(subh) == 0: 
            subh = [-1]       
        
        ## make the interpolation function; Quadratic as radial density drop is usually not linear
        ifunc = scipy.interpolate.interp1d(chianti_table['rat'][subh[0],:].flatten(),chianti_table['den'], kind="quadratic",fill_value="extrapolate")  
        
        ## apply the interpolation to the data 
        dens_1pix       = ifunc(rat_obs)                                                                                      
        #dens_1pix_noise = ifunc(rat_obs+rat_obs_noise) - dens_1pix
       
        ## debug prints
        #print("Radius from limb: ",rpos," at pixel positions (",xx,yy,")")         ## debug
        #print(len(subh),rpos/rsun)                                                 ## debug          
        return xx,yy,dens_1pix
###########################################################################
###########################################################################

