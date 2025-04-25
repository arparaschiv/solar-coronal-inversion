# -*- coding: utf-8 -*-

###############################################################################
### Parallel implementation of utilities for CLEDB_PREPINV                  ###
###############################################################################

#Contact: Alin Paraschiv, National Solar Observatory
#         arparaschiv@nso.edu
#         paraschiv.alinrazvan+cledb@gmail.com

### Tested Requirements  ################################################
### CLEDB database as configured in CLE V2.0.5; (db205 executable) or pycelp PyCELP commit b7678f8 with CHIANTI 10.1
# python                    3.13
# numpy                     2.1
# numba                     0.61
# llvmlite                  0.44 (as numba dependency probably does not need separate installing)
# scipy                     1.15
# astropy                   7.0
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
## Numba parralelization is now disabled by default. There are no reasons to enable it as runs are governed by multiprocessing.
## Only the code optimization functionality is preserved.

## Generally, if a numba non-python  (njit, or jit without explicit object mode flag)
## calls another function, the second should also be non-python compatible.

## Array indexing by lists and advanced indexing are generally not supported by numba.
## The np. calls belo are justa masks. All np. functions used are rewritten by numba.
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
from pylab import *
import scipy

import multiprocessing
from tqdm import tqdm
from astropy.wcs import WCS
import astropy.units as u
import time
import glob
import os
import sys

import ctrlparams
params=ctrlparams.ctrlparams()    ## just a shorter label

os.environ['NUMBA_DISABLE_JIT'] = str(np.int32(params.jitdisable))

from numba import jit,njit
from numba.typed import List  ## numba list is needed as standard reflected python list ingestion will be deprecated in numba

#import numexpr as ne ## Disabled as of update-iqud. No database compression going forward


#########################################################################
### Main Modules ########################################################
#########################################################################

###########################################################################
###########################################################################
#@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)  ### numba incompatible due to obs_headstructproc
def sobs_preprocess(sobs_in,headkeys,params):
    """
    Main function to process an input observation and ingest the relevant header keywords. Generally, this function iterates over the observation maps and sends each pixel to the internal functions. It returns a processed observation array (input dependent) that is ready for analysis. Additional products are calculated via its subfunctions. e.g. a height map (used to match databases), signal statistics, etc.

    Parameters:
    ----------
    sobs_in  : fltarr
               Input data array
    Headkeys : Input List of header keys, or record array, or structure
               Input list of header keys that describes the sobs_in data
    params   : class
               Passes a set of inversion controling parameters

    Returns
    -------
    sobs_tot    : fltarr
                  Contains the background subtracted, integrated, and normalized Stokes IQUV spectra for 1-line ([xs,ys,4]) or 2-line ([xs,ys,8]) observations.
    yobs        : fltarr
                  Used to construct a height projection for each observed voxel in units or R.
    sobs_totrot : fltarr
                  Derived from sobs_tot. The Stokes Q and U components are rotated along the center of the Sun to match the reference direction for linear polarization (the reference in which the database is created by CLEDB_BUILD). The variable is initialized as a “zero” array that is returned in the case of 1-line observations to keep a standardized function input/output needed for Numba vectorization.
    aobs        : fltarr
                  Stores the linear polarization angle transformation performed by the OBS_QUROTATE function. This information will be used to derotate the matched database profiles found by the CLEDB_INVPROC 2-line inversion function for comparison with the input Stokes profiles.
    dobs        : fltarr
                  Line ratio density array. This is used to match corresponding databases.
    snr         : fltarr
                  Returns the signal to noise ratio(SNR) of the root mean square (RMS) of the total counts in each Stokes profile. The SNR calculation is correspondent to the ratio between RMS intensity in the line core and background counts (the variance). This measurement shows the quality in the signal for a particular observed voxel.
    background  : fltarr
                  Returns averaged background counts for each observed voxel and each Stokes component.
    wlarr       : list of fltarr
                  List containing a set of wavelength dispersion axie for each observed line.
    keyvals     : list
                  list of parameters needed. These are read and processed from the observation metadata or processed depending on the input data type. Generated by the *sobs_preprocess*<--*obs_headstructproc* functions.
    issuemask   : fltarr
                  An array that encodes potential issues appearing during processing ONLY in the case of succesful processing. These are warnings, with respect to each pixel. Errors are handed by specific exceptions. This array will be updated across all modules. The tentative issuemask implementation is described separately.
    """

    if params.verbose >= 1:
        print('------------------------------------\n----SOBS_PREPROCESS - READ START----\n------------------------------------')
        if params.verbose >= 2:start0=time.time()

    ## if numba njit is enabled, remake the input list into a typed one.
    if params.jitdisable == False:
        sobs_in = List(sobs_in)                                   ## this is the List object of numba. It utilizes memory in a column-like fashion.

    ## unpack the minimal number of header keywords
    keyvals,hpxygrid,wlarr = obs_headstructproc(sobs_in,headkeys,params)
    if(keyvals == -1):
        if params.verbose >=1:
            print('SOBS_PREPROCESS: FATAL! OBS KEYWORD PROCESSING FAIL. Aborting!')
        return -1,0,0,0,0,0,0,0,0,0

    ##calculate an observation height and integrate the stokes profiles
    yobs=obs_calcheight(keyvals,hpxygrid)
    if params.integrated == False:
        if len(sobs_in[0].shape) == 4:
            sobs_tot,snr,background,issuemask = obs_integrate(sobs_in,wlarr,keyvals,params.verbose,params.ncpu,params.atmred)
        else:
            if params.verbose >=1:
                print('SOBS_PREPROCESS: FATAL! INPUT DATA does NOT have a wavelength dimension. Integrated keyword incorrect? Aborting!')
            return -1,0,0,0,0,0,0,0,0,0

    else:    ##Integrated observation array initialization
        sobs_tot   = np.concatenate((sobs_in[0],sobs_in[1]),axis=2)
        snr        = np.zeros((keyvals[0],keyvals[1],keyvals[3]*4),dtype=np.float32)
        background = np.zeros((keyvals[0],keyvals[1],keyvals[3]*4),dtype=np.float32)-1 ## set the array to -1 because it can't be computed
        issuemask  = np.zeros((keyvals[0],keyvals[1],keyvals[3]))

    if keyvals[3] == 2:          ##TWO LINE BRANCH

    ## for two-line inversions we require a rotation of the linear polarization Q and U components
    ## both variables are now initialized inside obs_qurotate
        sobs_totrot,aobs = obs_qurotate(sobs_tot,keyvals,hpxygrid)

    ## to efficiently match pycelp databases, we require an estimate of the apparent density of the plasma.
    ## A standalonve version of this function is available at: https://github.com/arparaschiv/FeXIII-coronal-density
    ## This function is currently available only for the Fe XIII pair (CLEDB always ratios 1074/1079) and requires a CHIANTI look-up table that is linked in this repository.
        if keyvals[4] == ["fe-xiii_1074","fe-xiii_1079"] or keyvals[4] == ["fe-xiii_1079","fe-xiii_1074"]:
            dobs,iss_temp = obs_dens(sobs_totrot,yobs,keyvals,params)
            issuemask[:,:,:]  += iss_temp[:,:,np.newaxis] ## density is one quantity for two input lines. If the lookup table height is insufficient this applies to both lines. Broadcasting the issue to each line.
        else:
            if params.verbose >=1:
                print('SOBS_PREPROCESS: Can NOT properly account for abundance between ion species.\nIncompatible with the PyCELP implementation.\nOnly CLE databases can theoretically be used with extreme caution (not recommended).')
            #return -1,0,0,0,0,0,0,0,0


        if params.verbose >=1:
            if params.verbose >= 2:
                print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL OBS PREPROCESS INTEGRATION AND ROTATION')
            print('------------------------------------\n--SOBS_PREPROCESS - READ FINALIZED--\n------------------------------------')

        return sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs

    else:                        ##ONE LINE BRANCH


        if params.verbose >=1:
            if params.verbose >= 2:
                print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL OBS PREPROCESS AND INTEGRATION')
            print('------------------------------------\n--SOBS_PREPROCESS - READ FINALIZED--\n-----------------------------------')

        return sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,np.zeros((sobs_tot.shape)),np.zeros((yobs.shape)),np.zeros((yobs.shape)) ## return the 0 arrays to keep returns consistent between 1 and 2 line inputs (it helps numba/jit).
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=params.jitparallel,forceobj=True,looplift=False,cache=params.jitcache)
def sdb_preprocess(yobs,dobs,keyvals,wlarr,params):
    """
    Main script to find and preload all necessary databases compatible with the sobs_preprocessed data.

    Returns databases as a list along with an encoding corresponding to each voxel of the observation.
    This is compatible to NUMBA object mode.

    Parameters:
    ----------
    yobs     : fltarr
             Array of observation heights of the shape of the input data array.
    dobs     : fltarr
             Array of computed observation plasma densities for matching with the database.
    keyvals : list
             Processed observation header keys.
    wlarr    : list of fltarr
             List containing a set of wavelength dispersion axie for each observed line.
    params   : class
             Passes a set of inversion controling parameters.

    Returns:
    -------
    dbnames     : strarr
                Array of entire database set file names to match to the observation.
    database    : fltarr
                Set of unique databases loaded into memory.
    dbhdr       : list
                header storing the configuration of the entire database set.
    db_enc_flnm : strarr            (output DEBUG/TEST only for verbose ==4)
                Array of filenames keeping track of which database applies to which pixel as resulting from *sdb_findelongationanddens* (same database can apply to multiple pixels).
    db_uniq     : list or strings   (output DEBUG/TEST only for verbose ==4)
                list that reduces the number of databases recorded in db_enc_flnm to help loading only the unique entries.
    db_enc      : fltarr            (output DEBUG/TEST only for verbose ==4)
                Array for each pixel that stores the index of the entry in db_uniq of respective db_enc_flnm entries to be uniquely loaded in the database variable.
    """

    if params.verbose >=1 and params.verbose <4:
        print('------------------------------------\n----SDB_PREPROCESS - READ START-----\n------------------------------------')
        if params.verbose >= 2 and params.verbose <4: start0=time.time()
    ######################################################################
    ## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx = keyvals[0]
    ny = keyvals[1]
    nline = keyvals[3]
    tline = keyvals[4]


    ######################################################################
    ## Preprocess the string information
    dbnames,dbynumbers,dbsubdirs=sdb_fileingest(params.dbdir,nline,tline,params.verbose)
    if params.verbose == 4:
        dbnames,dbynumbers,dbsubdirs=sdb_fileingest("/tests/dbfiles/",nline,tline,params.verbose)

    ## No database could be read? NO RUN!!!
    if dbsubdirs == 'Ingest Error':
        if params.verbose >=1 and params.verbose <4: print("SDB_PREPROCESS: FATAL! Could not read database.")
        return None,None,None

    ######################################################################
    ## Create a file encoding showing which database to read for which voxel
    ## the encoding labels at this stage do not have an order.
    db_enc=np.zeros((nx,ny),dtype=np.int32)         ## location in outputted database list; OUTPUT VARIABLE
    db_enc_flnm=np.zeros((nx,ny),dtype=np.int32)    ## location in file list when read from disk; used only internally in sdb_preprocess
    arg_array = []

    p         = multiprocessing.Pool(processes=min(params.ncpu, multiprocessing.cpu_count()-2),maxtasksperchild = 50000)     ## dynamically defined from system query as total CPU core number - 2

    for xx in range(nx):
        for yy in range(ny):
            arg_array.append((xx,yy,yobs[xx,yy],dobs[xx,yy],dbynumbers))

    if params.verbose >= 1 and params.verbose <4:       ## Some progressbar print output
        print("SDB_PREPROCESS: Matching optimal database based on elongation and density:")
        rs = p.starmap(sdb_findelongationanddens,tqdm(arg_array,total=len(arg_array)))

    else:                  ## No progressbar print output
        rs = p.starmap(sdb_findelongationanddens,arg_array)

    p.close()

    for i,res in enumerate(rs):
        xx,yy,db_enc_flnm[xx,yy] = res
    ##db_enc_flnm[xx,yy]=sdb_findelongationanddens(yobs[xx,yy],dobs[xx,yy],dbynumbers)

    ######################################################################
    ## makes a list of unique databases to read into memory. The full list will have repeated entries aa one database is assigned per pixel to solve;
    db_uniq=np.unique(db_enc_flnm)

    if params.verbose >= 1 and params.verbose <4:
        print("Available CLEDB databases cover a span of",len(np.unique(dbynumbers[:,1])),"solar heights between",dbynumbers[0,1]/100,"-",dbynumbers[-1,1]/100," radius")
        print("Load ",db_uniq.shape[0]," heights x densities  DB datafiles in memory for each of ",nline,"line(s).\n------------------------------------")

    ######################################################################
    ## read the required databases and their header
    ## The header and dimensions are passed as parameters to function calls to load the databases into memory.
    ## When ingesting multiple databases we assume all are of the same size.
    if nline == 2:                                            ## TWO-LINE BRANCH
        if dbsubdirs != "twolineDB":                          ## Main two-line branch
            dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
            database0=[None]*db_uniq.shape[0]
            for ii in range(db_uniq.shape[0]):
                database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=-1)
                for ij in range(7,-1,-1):
                    if dbhdr[-1] == 0:                        ## CLE database with density
                        database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
                    elif dbhdr[-1] == 1:                      ## PyCELP database with external density matching
                        ## Pycelp databases have Stokes V in units of signal per Angstrom. We divide the Stokes components if the input data is in signal per nm
                        ## Multiply by the conversion factor to scale the matched B field properly.
                        wvl_scl= 10 ** np.rint(np.log10((wlarr[0,0])/10000)) ## conversion factor between nm and angstrom measurements
                        if (ij ==7):
                            database0[ii][:,:,:,ij]=wvl_scl*database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
                        elif (ij ==3):
                            database0[ii][:,:,:,ij]=wvl_scl*database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
                        else:
                            database0[ii][:,:,:,ij]=database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
        # else:                                                ## Deprecated loop for older CLE databases that should not be used anymore. lines kept in case of need but commented/disabled.
        #     if params.verbose >= 1:
        #         print("A not common database generation scheme. Are you sure this is what you want to load?")
        #     dbhdr=[sdb_parseheader(params.dbdir+'db.hdr')][0] ## dont understand why the 0 indexing is needed...
        #     database0=[None]*db_uniq.shape[0]
        #     for ii in range(db_uniq.shape[0]):
        #         database0[ii]=sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose)
        #         for ij in range(7,-1,-1):
        #             if dbhdr[-1] == 0:
        #                 database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
        #             elif dbhdr[-1] == 1:
        #                 database0[ii][:,:,:,ij]=database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
    elif nline == 1:                                          ## ONE-LINE BRANCH ## Prepare for future, not really used downstream
        dbhdr=[sdb_parseheader(params.dbdir+dbsubdirs[0]+'db.hdr')][0]    ## assuming the same header info; reading the first DB header
        database0=[None]*db_uniq.shape[0]
        for ii in range(db_uniq.shape[0]):
            database0[ii]=np.append(sdb_read(dbnames[0][db_uniq[ii]],dbhdr,params.verbose),sdb_read(dbnames[1][db_uniq[ii]],dbhdr,params.verbose),axis=-1)
            for ij in range(3,-1,-1):
                if dbhdr[-1] == 0:                           ## CLE database with density
                    database0[ii][:,:,:,:,ij]=database0[ii][:,:,:,:,ij]/database0[ii][:,:,:,:,0]
                elif dbhdr[-1] == 1:                         ## PyCELP database with external density matching
                    ## Pycelp databases have Stokes V in units of signal per Angstrom. We divide the Stokes components if the input data is in signal per nm
                    ## Multiply by the conversion factor to scale the matched B field properly.
                    wvl_scl= 10 ** np.rint(np.log10(10000/(wlarr[0,0]))) ## conversion factor between nm and angstrom measurements
                    if (ij ==3):
                        database0[ii][:,:,:,ij]=wvl_scl*database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]
                    else:
                        database0[ii][:,:,:,ij]=database0[ii][:,:,:,ij]/database0[ii][:,:,:,0]

    ## reverted to use a list to feed the database set to the calculation as
    ## the numpy large array implementation does not parallelize properly leading to a 5x increase in runtime per 1024 calculations

    ######################################################################
    ## Create a numeric matched database encoding (db_enc) that is corresponding to indexable entries in the loaded database list
    for kk in range(db_uniq.shape[0]):
        db_enc[np.where(db_enc_flnm == db_uniq[kk])] = kk

    ## fix for standard reflected lists are deprecated as of numba 0.54;
    database = List()                            ## this is the List object implemented by NUMBA
    [database.append(x) for x in database0]

    ## [update issue mask] to implement

    if params.verbose >=1 and params.verbose <4:
        if params.verbose >= 2 and params.verbose <4:
            print("{:4.6f}".format(time.time()-start0),' SECONDS FOR TOTAL DB SEARCH AND FIND')
        print('------------------------------------\n--SDB_PREPROCESS - READ FINALIZED---\n------------------------------------')
    if params.verbose == 4:
        return db_enc,database,dbhdr,dbnames,db_enc_flnm,db_uniq ## extended return for debugging and unit testing. ok with numba object mode; not covered by normal verbosity levels
    else:
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
#@jit(parallel=params.jitparallel,forceobj=True,cache=params.jitcache)
def sdb_fileingest(dbdir,nline,tline,verbose):
    """
    Small function that returns filename and index of elongation y available in the full database set of calculations.
    """

    ##  This prepares the database directory files outside of numba non-python environment.
    ######################################################################
    ##  Read the directory structure and see what lines are available/computed in the database.
    linestr=["fe-xiii_1074/", "fe-xiii_1079/", "si-x_1430/", "si-ix_3934/", "mg-viii_3028/"]
    line_bin=[]
    dbsubdirs=[]
    for i in linestr:
        line_bin.append(os.path.isdir(dbdir+i))

    ######################################################################
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

        ## legacy loop to read two line stitched databases (for Fe XIII). .hdr and .DAT database files need to be in dbdir without a specific ion subfolder.
        ## Disabled as this is not an offered database building option anymore.
        # elif os.path.isfile(dbdir+"db.hdr") and glob.glob(dbdir+"DB*.DAT") != []:
        #     names=glob.glob(dbdir+"DB*.DAT")
        #     nn=str.find(names[0],'DB0')
        #     dbynumbers=np.empty(len(names),dtype=np.int32)
        #     for i in range(0,len(names)):
        #         dbynumbers[i]=np.int32(names[i][nn+2:nn+6])
        #     return [names,None],dbynumbers,'twolineDB' ##double return of names +none is superfluous; reason is to keep returns consistent regardless in terms of datatype of the if case

        elif sum(line_bin) ==1:
            if verbose >=2 and verbose <4: print("SDB_FILEINGEST: FATAL! Two line observation provided! Requires two individual ion databases in directory. Only one database is computed.")
            return [None,None],None,'Ingest Error'

        else:
            if verbose >=2 and verbose <4: print("SDB_FILEINGEST: FATAL! No database or incomplete calculations found in directory ")
            return [None,None],None,'Ingest Error'

    ######################################################################
    ## one line db prepare ???? is there a need for a database solution for just one line?
    ## sdb_preprocesss should not be called for one-line observations, as plasma and magnetic solutions can be computed analythically.
    elif nline ==1:
        if verbose >= 1 and verbose <4: print("Loadinng databases for one-line observations. Is this intended?")
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
            if verbose >=2 and verbose <4: print("SDB_FILEINGEST: FATAL! No database or incomplete calculations found in directory ")
            return [None,None],None,'Ingest Error'
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)               ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_findelongationanddens(xx,yy,y,d,dbynumbers):
    """
    Returns index of elongation y and density d in database set coresponding to the observed pixel.

    The calculation is faster ingested as a function rather than an inline calculation!
    """

    yyy = np.argwhere( np.abs( dbynumbers[:,1] / 100. - y ) == np.min(np.abs( dbynumbers[:,1] / 100. - y )) )  ## preselec for the observed height
    dd = np.argmin( np.abs( dbynumbers[yyy[0][0]:yyy[-1][0],2] / 100 - d ))                                     ## Out of yyy, select the closest matching density
    ii = yyy[0][0]+dd                                                                                          ## index is the sum of the starting height index and the density match.
    return xx,yy,np.int32(dbynumbers[ii,0])
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@jit(parallel=False,forceobj=True,cache=params.jitcache)  ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_parseheader(dbheaderfile):
    """
    Reads and parses the database header (db.hdr file) and returns the parameters contained in the set of calculations.

    db.hdr text file needs to have the specific CLE or PyCELP version format described in the CLEDB readme.
    """

    g=np.fromfile(dbheaderfile,dtype=np.float32,sep=' ')
    ## two kinds of data are returned
    ## 1. the linear coefficents of the form min, max, nx,theta,phi
    ## 2. logarithmically spaced parameters xed  (electron density array in CLE case); for these the min and max and number are not returned.

    ngy       = np.int32(g[0])             ## database file is individual with respect with ngy, no need to output further
    ned       = np.int32(g[1])             ## pycelp databases are individual also with respect to density. output only for consistency with cle
    ngx       = np.int32(g[2])
    nbphi     = np.int32(g[3])
    nbtheta   = np.int32(g[4])
    emin      = np.float32(g[5])
    emax      = np.float32(g[6])
    xed       = sdb_lcgrid(emin,emax,ned)    ## Ne is log. Unpack all computed densities using the min max and number of computed densities
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
    """
    Reading database files in memory and store in convenient variables.

    Returns
    -------
    db : fltarr [ncalc,nline,4]
       Output database file -1.590196830801
    """

    if verbose >=2 and verbose <4: start=time.time()

    ######################################################################
    ## unpack dbcgrid parameters from the database accompanying db.hdr file
    dbcgrid, ned, ngx, nbphi, nbtheta,  xed, gxmin,gxmax, bphimin, bphimax, \
    bthetamin, bthetamax, nline, wavel,dbtype  = dbhdr

    if verbose >=2 and verbose <4: print("INDIVIDUAL DB file location:", fildb)
    ##CLE database compresion introduced numerical instabilities at small values.
    ## It has been DISABLED in the CLE >=2.0.4 database building and CLEDB commit>=update-iqud
    #db=sdb_dcompress(np.fromfile(fildb, dtype=np.int16),verbose)

    ######################################################################
    ## load the CLE or PyCELP databases as is the case.
    if dbtype == 0:
        db = np.fromfile(fildb, dtype=np.float32)
    elif dbtype == 1:
        db = np.load(fildb)

    if verbose >=3 and verbose <4:
        print("SDB_READ: ",np.int64(sys.getsizeof(db)/1.e6)," MB in DB file")
        if np.int64(sys.getsizeof(db)/1.e6) > 250 : print("SDB_READ: WARNING! Very large DB. Lots of RAM are required. Processing will be slow!")

    if verbose >= 3 and verbose <4: print("{:4.6f}".format(time.time()-start),' SECONDS FOR INDIVIDUAL DB READ\n----------')

    if dbtype == 0:
        return np.reshape(db,(ned,ngx,nbphi,nbtheta,nline*4))
    elif dbtype == 1:
        return db
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=False,cache=params.jitcache)                              ## don't try to parallelize things that don't need as the overhead will slow everything down
def sdb_lcgrid(mmin,mmax,n):
    """
    Unpacks the grid for ned densities from mmin to mmax logarithms of density and n sampled densities.
    """

    ff=10.** ((mmax-mmin)/(n-1))
    xf=np.empty(n,np.float32)
    xf[0]=10.**mmin
    for k in range(1,n): xf[k]=xf[k-1]*ff
    return xf
###########################################################################
###########################################################################

###########################################################################
###########################################################################
##@jit(parallel=params.jitparallel,forceobj=True,looplift=True,cache=params.jitcache)  ## can not be numba object due to CoMP recarrays containing dtypes ##Functiona like hasattr are not non-python compatible.
def obs_headstructproc(sobs_in,headkeys,params):
    """
    Function for ingesting a minimal number of needed keywords from the observation headerfile.

    Most keywords are defined statically based on specs at time of last updating this function.
    Dummy headkeys input can be used for other instruments with excercising care as exampled in the MURAM and PSI cases.

    Implemented instruments are:
    a. CoMP/uCoMP/COSMO
    b. DKIST Cryo-NIRSP
    c. MURAM example simulation
    d. PSImas/pycelp example simulation
    e. CLE simulation
    """

    ## array dimensions!
    nx,ny=sobs_in[0].shape[0:2]
    #nx,ny=sobs_in[0].shape[0:2] if sobs_in[0].shape[0] == headkeys[0]['NAXIS1'] and sobs_in[0].shape[1] == headkeys[0]['NAXIS2'] else print("Mismatch in array shapes; Fix before continuing") ## alternatively, these can be read from naxis[1-3]; Will be updated when keywords are known for all sources

    
    ######################################################################
    ## Unpacks the header metadata information from the observation
    linestr = ["fe-xiii_1074","fe-xiii_1079","si-x_1430","si-ix_3934", "mg-viii_3028/"]
    if len(sobs_in) == 2:
        nline = 2

        ######################################################################
        ## Cryo scans might not have spatial coordinates that fit into a rectangular coordinate grid; e.g. rotated maps. A grid map needs to be created in this case.
        ## This map stays zero for any other data sources where coordinates can be a rectangular box
        hpxygrid=np.zeros((2,nx,ny),dtype=np.float32)

        if params.integrated == True:
            ############ for CoMP ############################################
            if ('WAVETYPE' in headkeys[0].keys()) and ('WAVETYPE' in headkeys[1].keys()):
                tline=[[i for i in linestr if headkeys[0]['WAVETYPE'] in i][0],[i for i in linestr if headkeys[1]['WAVETYPE'] in i][0]]
                ## searches for the "wavetype" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed
                ## the [0] unpacks the one element lists where used.
            ############ for uCoMP  ##########################################
            elif ('FILTER' in headkeys[0].keys()) and ('FILTER' in headkeys[1].keys()):
                tline=[[i for i in linestr if headkeys[0]['FILTER'] in i][0],[i for i in linestr if headkeys[1]['FILTER'] in i][0]]
                ## searches for the "FILTER" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed
                ## the [0] unpacks the one element lists where used.
            ############ for DKIST Cryo-NIRSP integrated data  ###############
            elif ('LINEWAV' in headkeys[0].keys()) and ('LINEWAV' in headkeys[1].keys()):
                try:
                    tline=[[i for i in linestr if str(headkeys[0]['LINEWAV'])[:4] in i][0],[i for i in linestr if str(headkeys[1]['LINEWAV'])[:4] in i][0]]
                except:
                    tline=[[i for i in linestr if str(headkeys[0][0]['LINEWAV'])[:4] in i][0],[i for i in linestr if str(headkeys[1][0]['LINEWAV'])[:4] in i][0]]
                ## searches for the "LINEWAV" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed
                ## the [0] unpacks the one element lists where used.
                ## extra1: The LINEWAV is a float, we need to convert it back to a string
                ## extra2: if the metadata from the header is the list from the asdf even if integrated data is used, we need to pick just one slit position. [0][0] and [0][1] selections.

                if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: WARNING! Is this Cryo-NIRSP data processed into integrated quantities?.")
            ############ Cant understand what this data is  ##################
            else:
                if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: FATAL! 2-line integrated? Can't read line information from keywords.")
                return -1     #catastrophic exit

        ## SPECTROSCOPIC DATA TO BE INTEGRATED
        ## CLE-SIM, MURAM-SIM, and Cryo-NIRSP 1074/1079 ratios are implemented via the LINEWAV keyword
        ## Other ratios require scientific validation before using
        else:
            if ('LINEWAV' in headkeys[0].keys()) and ('LINEWAV' in headkeys[1].keys()):
                try:
                    tline=[[i for i in linestr if str(headkeys[0]['LINEWAV'])[:4] in i][0],[i for i in linestr if str(headkeys[1]['LINEWAV'])[:4] in i][0]]
                except:
                    tline=[[i for i in linestr if str(headkeys[0][0]['LINEWAV'])[:4] in i][0],[i for i in linestr if str(headkeys[1][0]['LINEWAV'])[:4] in i][0]]
                ## searches for the "LINEWAV" substring in the linestr list of available lines.
                ## the headkey is a bytes type so conversion to UTF-8 is needed
                ## the [0] unpacks the one element lists where used.
                ## extra1: The LINEWAV is a float, we need to convert it back to a string
            else:
                if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: FATAL! 2-line Spectroscopy; Can't read line information from keywords.")
                return -1     #catastrophic exit
            if tline != ['fe-xiii_1074', 'fe-xiii_1079']:
                if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: FATAL! 2-line Spectroscopy; Lines are not 1074 and 1079 (in that order). Other ratios are not implemented yet! Aborting.")
                return -1     #catastrophic exit

    ## 1-LINE BRANCH
    ## Not important if integrated here
    if len(sobs_in) == 1:
        nline=1

    ######################################################################
        ## Cryo scans might not have spatial coordinates that fit into a rectangular coordinate grid; e.g. rotated maps. A grid map needs to be created in this case.
        ## This map stays zero for any other data sources where coordinates can be a rectangular box
        hpxygrid=np.zeros((1,nx,ny),dtype=np.float32)

        ############ for CoMP  ###########################################
        if 'WAVETYPE' in headkeys[0].keys():
            tline=[[i for i in linestr if headkeys['WAVETYPE'] in i][0],0] ## explanation in blocks above
        ############ for uCoMP  ##########################################
        elif 'FILTER' in headkeys[0].keys():
            tline=[[i for i in linestr if headkeys['FILTER'] in i][0],0] ## explanation in blocks above
        ############ for DKIST Cryo-NIRSP or simulations #################
        elif 'LINEWAV' in headkeys[0].keys():
            try:
                tline=[[i for i in linestr if str(headkeys[0]['LINEWAV'])[:4] in i][0],0] ## explanation in blocks above
            except:
                tline=[[i for i in linestr if str(headkeys[0][0]['LINEWAV'])[:4] in i][0],0] ## explanation in blocks above
        ## 0 added to the list for proper processing of the lines in cledb_proc. it is not used.
        ## one line setup for iuqd CoMP/uCoMP/COSMO data not implemented

    ## check if nline is correct. the downstream inversion modules are contingent on this keyword
    if nline != 1 and nline != 2:
        if params.verbose >= 2: print("OBS_HEADSTRUCTPROC: FATAL! Not a one-line or two-line observation. Aborting.")
        return -1        #catastrophic exit

    if params.verbose >= 1:
        print('Inverting observations of',nline,'coronal line(s) ')
        if nline >= 1:
            if   tline[0] == linestr[0]:
                print('Line 1: Fe XIII 1074.7nm')
            elif tline[0] == linestr[1]:
                print('Line 1: Fe XIII 1079.8nm')
            elif tline[0] == linestr[2]:
                print('Line 1: Si X    1430.1nm')
            elif tline[0] == linestr[3]:
                print('Line 1: Si IX   3934.3nm')
            elif tline[0] == linestr[4]:
                print('Line 1: Mg VIII   3028.1nm')
            if nline == 2:
                if   tline[1] == linestr[0]:
                    print('Line 2: Fe XIII 1074.7nm')
                elif tline[1] == linestr[1]:
                    print('Line 2: Fe XIII 1079.8nm')
                elif tline[1] == linestr[2]:
                    print('Line 2: Si X    1430.1nm')
                elif tline[1] == linestr[3]:
                    print('Line 2: Si IX   3934.3nm')
                elif tline[1] == linestr[4]:
                    print('Line 2: Mg VIII   3028.1nm')


    if params.integrated != True:
        nw=sobs_in[0].shape[2]
        #nw=sobs_in[0].shape[2] if sobs_in[0].shape[2] == headkeys[0]['NAXIS3'] else print("Mismatch in array shapes; Fix before continuing")
    else:
        nw=1                        ## For line-integrated data; dont set 0 or arrays cant initialize
    ## Define a wavelength dispersion array that transports the dispersion axis.
    wlarr = np.zeros((nw,nline),dtype=np.float32)

## array coordinate keywords
## These needs to be changed based on observation. For CLE this is normally the information in GRID.DAT
    if headkeys[0]['INSTRUME'] == "CLE-SIM" or headkeys[0]['INSTRUME']  == "MUR-SIM"\
    or headkeys[0]['INSTRUME'] == "PyCELP":                                                   ## CASE 1: Simulated OBSERVATIONS
        crpix1 = headkeys[0]['CRPIX1']                                                        ## Rerefence pixels along all dimensions
        crpix2 = headkeys[0]['CRPIX2']
        if nline == 1:
            crpix3 =  [np.int32(headkeys[0]['CRPIX3']),0]                                       ## Dummy 0 output, not used.
        else:
            crpix3 = [np.int32(headkeys[0]['CRPIX3']),np.int32(headkeys[1]['CRPIX1'])]        ## two lines have different wavelength parameters

        crval1 = np.float32(headkeys[0]['CRVAL1'])                                            ## solar/wavelength coordinates at crpixn in r_sun/angstrom;
        crval2 = np.float32(headkeys[0]['CRVAL2'])
        if nline == 1:
            crval3 = [np.float32(headkeys[0]['CRVAL3']),0]                                     ## Dummy 0 output, not used.
        else:
            crval3 = [np.float32(headkeys[0]['CRVAL3']), np.float32(headkeys[1]['CRVAL3'])]

        cdelt1 = np.float32(headkeys[0]['CDELT1'])                                            ## spatial/spectral resolution in R_sun/angstrom
        cdelt2 = np.float32(headkeys[0]['CDELT2'])
        if nline == 1:
            cdelt3 = [np.float32(headkeys[0]['CDELT3']),0]                                    ## dummy 0 output
        else:
            cdelt3 = [np.float32(headkeys[0]['CDELT3']), np.float32(headkeys[1]['CDELT3'])]

    elif (headkeys[0]['INSTRUME'] == "COMP"):                                                 ## CASE 2: COMP OBSERVATIONS

        crpix1 = headkeys[0]['CRPIX1']                                                        ## Comp takes reference at center
        crpix2 = headkeys[0]['CRPIX2']
        crpix3 = [0,0]                                                                        ## not used here ## two lines have different wavelength parameters

        crval1 = np.float32(headkeys[0]['CRVAL1']*720/695700.0)                               ## solar coordinates at crpixn in r_sun; from -310.5*4.46 ##arcsec to R_sun conversion via 720/695700
        crval2 = np.float32(headkeys[0]['CRVAL2']*720/695700.0)
        if nline == 1:
            crval3 = [np.float32(headkeys[0]['WAVE_REF']),0]                                   ## Dummy 0 output, not used.
        else:
            crval3 = [np.float32(headkeys[0]['WAVE_REF']), np.float32(headkeys[1]['WAVE_REF'])]   ## not really used

        cdelt1 = np.float32(headkeys[0]['CDELT1']*720/695700.0)                               ## COMP Cdelt in R_sun
        cdelt2 = np.float32(headkeys[0]['CDELT2']*720/695700.0)
        if nline == 1:
            cdelt3 = [0.0,0.0]                                                                ## Dummy 0 output, not used.
        else:
            cdelt3 = [0.0,0.0]                                                                ## not used for integrated data such as CoMP

    elif (headkeys[0]['INSTRUME'] == "UCoMP"):                                                ##CASE 3: uCoMP OBSERVATIONS;;

        crpix1 = headkeys[0]['CRPIX1']                                                        ## Comp takes reference at center
        crpix2 = headkeys[0]['CRPIX2']
        if nline == 1:
            crpix3 = [0,0]                                                                    ## Dummy 0 output, not used.
        else:
            crpix3 = [0,0]                                                                    ## not used here ## two lines have different wavelength parameters

        crval1 = np.float32(headkeys[0]['CRVAL1']*720/695700.0)                               ## solar coordinates at crpixn in r_sun; from -310.5*4.46 ##arcsec to R_sun conversion via 720/695700
        crval2 = np.float32(headkeys[0]['CRVAL2']*720/695700.0)
        if nline == 1:
            crval3 = [np.float32(headkeys[0]['FILTER']),0]                                     ## Dummy 0 output, not used.
        else:
            crval3 = [np.float32(headkeys[0]['FILTER']), np.float32(headkeys[1]['FILTER'])]   ## not really used

        cdelt1 = np.float32(headkeys[0]['CDELT1']*720/695700.0)                               ## COMP Cdelt in R_sun
        cdelt2 = np.float32(headkeys[0]['CDELT1']*720/695700.0)
        if nline == 1:
            cdelt3=[0.0,0.0]                                                                  ## Dummy 0 output, not used.
        else:
            cdelt3 = [0.0,0.0]

    elif (headkeys[0][0]['INSTRUME'] == "CRYO-NIRSP") or (headkeys[0]['INSTRUME'] == "CRYO-NIRSP"):  ## CASE 3: Cryo-NIRSP OBSERVATIONS;
        ### Cryo-NIRSP specifications defines CRPIX1=wavelength, CRPIX2= spatial and CRPIX3=spatial.
        ### We translate these to keep all data sources consistent to: CRPIX1=spatial, CRPIX2= spatial and CRPIX3=wavelength


        ########### GET SPECTRAL DISPERSION AXIS #######
        ## Use the WCS tools for this as the dispersion axis in not strictly linear
        ## the spectral dispersion axis type (CTYPE1) is 'AWAV-GRA' refers to a grating func for air wavelengths
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  ## TO ELIMINATE datafix warnings
            wcs1 = WCS(headkeys[0][0])
            nwv1 = headkeys[0][0]['NAXIS1']
            wlarr[:,0] = wcs1.array_index_to_world(0,0,np.arange(nwv1))[0].to(u.nm).value[:2].reshape(1,nwv1)
            if nline == 2:
                wcs2 = WCS(headkeys[1][0])
                nwv2 = headkeys[1][0]['NAXIS1']
                wlarr[:,1] = wcs2.array_index_to_world(0,0,np.arange(nwv2))[0].to(u.nm).value[:2]


        crpix1 = headkeys[0][0]['CRPIX2']                                    ##
        crpix2 = headkeys[0][0]['CRPIX3']
        crpix3 = [0,0]                                                       ## use the reference at pixel 0

        crval1 = np.float32(headkeys[0][0]['CRVAL2']*720/695700.0)           ## solar coordinates at crpixn in r_sun; from -310.5*4.46 ##arcsec to R_sun conversion via 720/695700
        crval2 = np.float32(headkeys[0][0]['CRVAL3']*720/695700.0)
        crval3 = [wlarr[0,0], wlarr[0,1]]                                    ## Left wavelength position of the list.

        cdelt1 = np.float32(headkeys[0][0]['CDELT2']*720/695700.0)           ## Cdelt in R_sun
        cdelt2 = np.float32(headkeys[0][0]['CDELT3']*720/695700.0)
        cdelt3 = [wlarr[1,0] - wlarr[0,0], wlarr[1,1] - wlarr[0,1]]          ## WCS calculated dispersion. This might be slightly different than the header value. Assumes this is linearized


        ## update the hpxygrid with the scans. We calculate only for the first map
        p         = multiprocessing.Pool(processes=min(params.ncpu, multiprocessing.cpu_count()-2),maxtasksperchild = 50000)     ## dynamically defined from system query as total CPU core number - 2
        arg_array = []

        for yy in range(ny):
            arg_array.append((yy,headkeys[0][ny],WCS(headkeys[0][yy]),headkeys[0][yy]["NAXIS2"]))

        if params.verbose >= 1:       ## Some progressbar print output
            print("OBS_HEADSTRUCTPROC: Processing the pointing and header information:")
            rs = p.starmap(obs_headstructproc_work_1pix,tqdm(arg_array,total=len(arg_array)))

        else:                  ## No progressbar print output
            rs = p.starmap(obs_headstructproc_work_1pix,arg_array)

        p.close()

        for i,res in enumerate(rs):
            xloc,yloc,x,y = res
            hpxygrid[0,:,yloc] =  x
            hpxygrid[1,:,yloc] =  y

    else:
        if params.verbose >= 1: print("OBS_HEADSTRUCTPROC: FATAL! Observation keywords not recognized.") ## fatal error
        if params.verbose >= 2: print("only Cryo-NIRSP, CoMP, and uCoMP data and the MURAM and CLE simulations are currently implemented as shown in examples.")

    ## create a wavelength array based on keywords for each line to be processed if that does not exist from upstream preprocessing
    if np.any(wlarr == 0):
        for i in range(nline):
            if crpix3[i] == 0:                                         ## simple; just cycle and update
                for j in range(nw):
                    wlarr[j,i] = crval3[i]+(j*cdelt3[i])
            else:                                                      ## If reference pixel is not 0, cycle forwards and backwards from crpix3
                for j in range(crpix3[i],nw):                          ## forward cycle
                    wlarr[j,i] = crval3[i]+((j-crpix3[i])*cdelt3[i])
                for j in range(crpix3[i],-1,-1):                       ## Backwards cycle; -1 end index to fill the 0 index of the array
                    wlarr[j,i] = crval3[i]-((crpix3[i]-j)*cdelt3[i])


## Additional keywords of importance that might or might not be included in observations.
## assign the database reference direction of linear polarization. See CLE routine db.f line 120.
## Angle direction is trigonometric; values are in radians
## 0 for horizontal ->0deg; np.pi/2 for vertical ->90deg rotation in linear polarization QU.
    linpolref = params.dblinpolref ##  0 is the reference used in paraschiv & Judge 2022 for the direction used in computing the database (at Z=0 plane).

## instrumental line broadening/width should be read and quantified here
## not clear at this point if this will be a constant or a varying keyword
## a 0 value will skip including an instrumental contribution to computing quick look non-thermal widths
    instwidth = params.instwidth

    ## pack the decoded keywords into a comfortable python list variable to feed to downstream functions/modules
    return [nx,ny,nw,nline,tline,crpix1,crpix2,crpix3,crval1,crval2,crval3,cdelt1,cdelt2,cdelt3,linpolref,instwidth],hpxygrid,wlarr
###########################################################################
###########################################################################

###########################################################################
###########################################################################
def obs_headstructproc_work_1pix(yy,wcs_pix,ys):
    """
    Helper function defining for parallelizing *obs_headstructproc*.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  ## TO ELIMINATE datafix warnings
        xy = wcs_pix.array_index_to_world(0,np.arange(ys),0)[1]
        x,y,obstime = xy.Tx.value,xy.Ty.value
    ## return current scan step number, current measurement number,
    ## Tx and Ty helioprojective coordinates, and the observation time.
    return yy,x,y
###########################################################################
###########################################################################


###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_calcheight(keyvals,hpxygrid):
    """
    Helper function for observation preprocessing in *obs_headstructproc* --- Calculates the off-limb heights corresponding to each voxel.

    Assumes that both observations have the same pointing.
    """

    ######################################################################
    ## Unpack the required keywords These are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.
    nx,ny = keyvals[0:2]
    crpix1,crpix2=keyvals[5:7]
    crval1,crval2=keyvals[8:10]
    cdelt1,cdelt2=keyvals[11:13]

    if crpix1 != 0:
        crval1 = crval1 - (crpix1 * cdelt1)
    if crpix2 != 0:
        crval2 = crval2 - (crpix2 * cdelt2)

    yobs=np.empty((nx,ny),dtype=np.float32)

    ## Check for custom hpxy grid of coordinates for Cryo-NIRSP data
    if np.sum(hpxygrid) == 0.:   ## Other data sources
        for xx in range(nx):
            for yy in range(ny):
                yobs[xx,yy] = np.sqrt((crval1+(xx*cdelt1))**2+(crval2+(yy*cdelt2))**2 ) ## use just the pointing keywords
    else:                        ## Cryo_NIRSP
        for xx in range(nx):
            for yy in range(ny):
                yobs[xx,yy] = np.sqrt(hpxygrid[0,xx,yy]**2 + hpxygrid[1,xx,yy]**2 )     ## Use the rectangular coordinate grid array

    return yobs
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_integrate(sobs_in,wlarr,keyvals,verbose,ncpu,atmred):
    """
    Reads the background and peak emission of lines, subtracts the background and integrates the signal.

    Function was slow for big arrays. It has been rewritten for efficiency, but lost a lot in terms of clearness.
    Extended comments are provided for each step/component. Versions prior to "update-iqud" commit contain the original implementation.

    Sobs_in is an array of spatial size, wavelength size and 4 components, or a list of two such arrays for two-line obs.
    The lines are subscribed via ZZ. The last dimension is always 4 corresponding to Stokes IQUV/
    the output arrays are of last dimension 4 or 8 based on input. These are subscribed via iterating ZZ; 0:4 or 4:8 via (4*zz):(4*(zz+1))
    """

    ######################################################################
    ## load what is needed from keyvals (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny,nw = keyvals[0:3]
    nline    = keyvals[3]


    sobs_tot          = np.zeros((nx,ny,nline*4),dtype=np.float32)      ## output array of integrated profiles
    background        = np.zeros((nx,ny,nline*4),dtype=np.float32)      ## output array to record the background levels in stokes I,Q,U,V, in that order
    snr               = np.ones((nx,ny,nline*4),dtype=np.float32)       ## output array to record snr statistics of the profile
    issuemask_obsint  = np.zeros((nx,ny,nline),dtype=np.int32)           ## mask to test the statistical significance of the data: mask[0]=1 -> no signal above background in stokes I; mask[1]=1 -> distribution is not normal (skewed); mask[2]=1 line center (observed) not found; mask[3]=1=2 one or two fwhm edges were not found

    ######################################################################
    ## integrate for total profiles.

    ## set up the cpu worker and argument arrays
    p         = multiprocessing.Pool(processes=min(ncpu, multiprocessing.cpu_count()-2),maxtasksperchild = 50000)     ## dynamically defined from system query as total CPU core number - 2
    ## argument index keeper for splitting tasks to cpu cores
    arg_array = []

            ## WHICH CORONAL LINE IS IN THE DATA ?
        ## Currently we handle the two Fe XIII lines in wavelength order. Other cases do not have well established and validated atlas approaches YET!
    wvc=[]
    wCont=[]
    for zz in range(nline):
        if (1074.65 > wlarr[:,zz].min() and 1074.65 < wlarr[:,zz].max()):
            wvc.append(1074.65)
            wCont.append(np.argmin(np.abs(wlarr[:,zz] - 1074.)))
        elif (1079.8 > wlarr[:,zz].min() and 1079.8 < wlarr[:,zz].max()):
            wvc.append(1079.8)
            wCont.append(np.argmin(np.abs(wlarr[:,zz] - 1080.2)))

    if len(wvc) == 0 or len(wCont) == 0:
        if verbose >= 1: print("OBS_INTEGRATE: FATAL! Wavelength case not handled for atlas fitting!")
        return -1,-1,-1


    # if nline == 2:
    #     wCont = [np.argmin(np.abs(wlarr[:,0] - 1074.)),  np.argmin(np.abs(wlarr[:,1] - 1080.2))]   ## pixel location of clean continuum
    # else:
    #     wCont = [np.argmin(np.abs(wlarr[:,0] - 1074.))]

    if atmred == True:                 ## Flag to reduce the atmosphere from cryo-NIRSP data


        ## DOWNLOAD THE ATLAS FILES IS NOT ALREADY IN DIRECTORY
        if not os.path.exists('./CLEDB_PREPINV/atlases/solar_merged_20240731_600_33300_000.out.gz'):
            os.makedirs("./CLEDB_PREPINV/atlases/", exist_ok=True)
            os.chdir('./CLEDB_PREPINV/atlases/')
            os.system("wget -nc --no-parent https://github.com/tschad/dkist_telluric_atlas/raw/main/atlases/telluric_atlas_mainMol_USstd_wv_air_angstrom_v20240307.npy")
            os.system("wget -nc --no-parent https://github.com/tschad/dkist_telluric_atlas/raw/main/atlases/telluric_atlas_mainMol_USstd_CO2_416ppm-Base_3km-PWV_3__mm-Airmass_1___v20240307.npy")
            os.system("wget -nc --no-parent https://mark4sun.jpl.nasa.gov/toon/solar/solar_merged_20240731_600_33300_000.out.gz")
            os.chdir('../../')

        ## LOAD THE TELLURIC ATLASES
        wvair_telluric  = np.load('./CLEDB_PREPINV/atlases/telluric_atlas_mainMol_USstd_wv_air_angstrom_v20240307.npy')  /10.                  ## converting A to nm
        trans_telluric  = np.load('./CLEDB_PREPINV/atlases/telluric_atlas_mainMol_USstd_CO2_416ppm-Base_3km-PWV_3__mm-Airmass_1___v20240307.npy')

        ## LOAD SOLAR ATLAS           ## data is in wavenumber [cm^-1] in vacuum, it needs to be in wavelength in air.
        dc   = np.loadtxt('./CLEDB_PREPINV/atlases/solar_merged_20240731_600_33300_000.out.gz',skiprows =3)  ## disk integrated  -- 100
        wvair_solar = vac2air(1e8/dc[:,0])[::-1] ## Converts vac to air
        atlas_solar = dc[:,1][::-1]

        ## Bounds of the model variables; default values
        bounds = [ (30000,70000),                                                               ## Resolving power of line spread function
                   (0.05,2),                                                                    ## telluric opacity factor
                   (0,3),                                                                       ## velocity shift in km/s for the solar atlas  [TUNED ACCORDING TO PRELIMINARY FITS]
                   (-3,0),                                                                      ## velocity shift in km/s for the telluric atlas [TUNED ACCORDING TO PRELIMINARY FITS]
                   (0.,0.5),                                                                    ## Spectrograph constant straylight fraction
                   (0.95*sobs_in[0][0,0,wCont[0],0],1.05*sobs_in[0][0,0,wCont[0],0]) ]         ## Amplitude of the background scattered light in millionths

        for zz in range(nline): ## to travel through the two lines
            ## INTERPOLATE ATLASES ONTO THE SPECTRAL AXIS OF THE CRYONIRSP DATA
            trans_telluric = np.interp(wlarr[:,zz],wvair_telluric,trans_telluric)
            atlas_solar    = np.interp(wlarr[:,zz],wvair_solar,atlas_solar)

            ## Spectrograph observational parameters and fitting weightsos.environ['NUMBA_DISABLE_JIT'] = '1'
            wv_mean = wlarr[:,zz].mean()                   ## mean value in the spectral axis
            dwv = np.mean(np.abs(wlarr[1:,zz]-wlarr[:-1,zz]))  ## spectral resolution in the observation, assuming constant dispersion, one float value.
            ##wldiff = np.diff(wlarr[:,zz])                   ## spectral resolution in the observation. Array at each sampling point
            ## Define spectral weights to apply
            Iwgts = np.zeroes_like(nw)

            ## ...set weights outside of coronal region
            Iwgts[(wlarr[:,zz]<(wvc[zz]-0.5))*(wlarr[:,zz]>(wvc[zz]+0.5))] = 1.  ## Try fitting the main photospheric or telluric lines regions
            Iwgts[wlarr[:,zz]>(wlarr[:,zz][-1]-0.5)]                       = 0   ## Ignore the field-stop right edge of the spectra where intensities are non-linear

            for xx in range(nx):
                for yy in range(ny):
                    bounds[-1] = (0.95*sobs_in[zz][xx,yy,wCont[zz],0],1.05*sobs_in[zz][xx,yy,wCont[zz],0])  ##update bounds for individual pixel
                    ## We fit only Stokes I. Currently we do not attempt resolving cross-talk nor polarization of photospheric lines.
                    res = differential_evolution(objFuncStokesI,bounds,args = (sobs_in[zz][xx,yy,:,0],),disp = True,tol=1.e-10,maxiter = 75,
                                                 popsize = 1,polish = True,callback=callBFunc)
                    ## Calculate fitted line profile and subtract it from the data
                    arg_array.append((xx,yy,zz,nw,wlarr[:,zz],wCont[zz],dwv,sobs_in[zz][xx,yy,:,:] - modelStokesI(res.x)))

    else:  ## assuming the spectral profile is reduced to just the coronal contribution
        for zz in range(nline): ## to travel through the two lines
            ## Feed the wavelength array directly but also the difference to not compute gradiantes for each pixel.
            #wldiff = np.diff(wlarr[:,zz])       ## spectral resolution in the observation
            dwv    = np.mean(np.diff(wlarr[:,zz]))
            for xx in range(nx):
                for yy in range(ny):
                    arg_array.append((xx,yy,zz,nw,wlarr[:,zz],wCont[zz],dwv,sobs_in[zz][xx,yy,:,:]))

    if verbose >= 1:       ## Some progressbar print output
        print("OBS_INTEGRATE: Integrating the Stokes IQUV spectra:")
        rs = p.starmap(obs_integrate_work_1pix,tqdm(arg_array,total=len(arg_array)))

    else:                  ## No progressbar print output\
        rs = p.starmap(obs_integrate_work_1pix,arg_array)

    p.close()

    for i,res in enumerate(rs):
        xx,yy,zz,sobs_tot[xx,yy,(4*zz):(4*(zz+1))],snr[xx,yy,(4*zz):(4*(zz+1))],background[xx,yy,(4*zz):(4*(zz+1))],issuemask_obsint[xx,yy,zz] = res

    return sobs_tot,snr,background,issuemask_obsint
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(fastmath=True)
def objFuncStokesI(para,ii):
    """
    Objective function  for *obs_integrate_atmred_1pix* -- Reduced Chi-Squared Statistic.
    """

    ifit = obs_integrate_atmred_1pix(para)
    ires = np.sum((ii-ifit)**2  * Iwgts) / np.sum(Iwgts)
    return ires
###########################################################################


@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_integrate_atmred_1pix(para):
    """
    Worker function of *obs_integrate* --- used to reduce telluric and photospheric lines from spectra.

    Calculates a forward model of spectral line observation assuming single Gaussian profile for coronal line.
    This function and its helpers are derivatives of the Cryo-NIRSP community data redcution and analysis tutorial: https://bitbucket.org/dkist-community-code/cryonirsp-notebooks/src/main/first_release_specFitting/

    Parameters
    ----------
    LSF_RPOW  : Resolving power of line spread function
    opac      : An opacity factor to modify total telluric line absorpance
    velS      : Velocity shift of solar atlas relative to observed spectrum
    velT      : Velocity shift of telluic atlas relative to observed spectrum
    strayfrac : Additional scalar fraction of straylight within spectrograph (i.e. the "veil" component)
    icont     : Amplitude of the background continuum intensity (dominated by scattered light)

    Returns
    -------
    ifit      : A model fit of the observation
    """

    LSF_RPOW,opac,velS,velT,strayfrac,icont,= para

    ## telluric transmission and scattered photospheric light contributions
    ftsTmod = np.interp(wlarr1,wlarr1 + velT/3e5*wv_mean,trans_telluric**opac)
    ftsSmod = np.interp(wlarr1,wlarr1 + velS/3e5*wv_mean,atlas_solar)
    ftsmod  = ftsTmod * ftsSmod

    ## add straylight and scale for amplitude
    ifit = icont * (ftsmod + strayfrac) / (1. + strayfrac)

    ## Gaussian convolution of the FTS atlast
    kern_pix  = (wvc / LSF_RPOW) / 2.35482 / dwv
    ifit = gaussian_filter1d_numba(ifit, kern_pix)

    return ifit
###########################################################################

def callBFunc(intermediate_result):
    """
    Callback function for *obs_integrate_atmred_1pix* --- stop iterations once objective function is <0.3.
    """
    return intermediate_result.fun < 0.3
###########################################################################

def vac2air(wave_vac):
    """
    Helper function for *obs_integrate* --- Converts wavelengths from vacuum to air-equivalent when loading fitting atlases.
    """
    wave_air = np.copy(wave_vac)
    ww  = (wave_vac >= 2000)
    sigma2 = (1e4 / wave_vac[ww])**2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - sigma2) + 0.00015998 / (38.9 - sigma2)
    wave_air[ww] = wave_vac[ww] / n / 10. ## last division converts A to nm
    return wave_air
###########################################################################

@njit(parallel=params.jitparallel,cache=params.jitcache,fastmath=True)
def gaussian_filter1d_numba(input_array, sigma):
    """
    Helper function for *obs_integrate* --- Applies a Gaussian 1D convolution to 1D array 
    """

    # Generate Gaussian kernel
    radius = int(3 * sigma + 0.5)  ## USING ONLY 3 STD DEVS FOR SPEED
    x = np.arange(-radius, radius + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()
    output = np.zeros_like(input_array)
    for i in range(radius,len(input_array)-radius):
        for k in range(len(gaussian_kernel)):
            j = i + k - radius
            output[i] += input_array[j] * gaussian_kernel[k]
    ## edge treatment
    output[:radius] = input_array[:radius]
    output[(-radius):] = input_array[(-radius):]
    return output
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_integrate_work_1pix(xx,yy,zz,nw,wlarr,wcont,dwv,sobs_in_1pix):
    """
    Worker function for parallelizing *obs_integrate* tasks.
    """

    if ((np.count_nonzero(sobs_in_1pix[:,0])) and (np.isfinite(sobs_in_1pix[:,:]).all())):  ## enter only for pixels that have counts recorded
        ## cdf method works for noisier data but hard to get the limits to work generally. 
        ## We changed the implementation to just evaluating the continuum in regions not affected by absorptions.
        #compute the rough cdf distribution of the stokes I component to get a measure of the statistical noise.

        cdf = obs_cdf(sobs_in_1pix[:,0]) ## keep CDF as a full array

        ## compute the 1%-99% distributions to measure the quiet noise.
        #l0 =np.argwhere(np.abs(cdf[xx,yy,:]-0.01) == np.min(np.abs(cdf[xx,yy,:]-0.01)))[-1][0]
        #r0=np.argwhere(np.abs(cdf[xx,yy,:]-0.99) == np.min(np.abs(cdf[xx,yy,:]-0.99)))[-1][0]
        #l0=np.argwhere(np.abs(cdf[xx,yy,:]) >= 0.01)[0][0]  ## easier on the function calls than original version above
        #r0=np.argwhere(np.abs(cdf[xx,yy,:]) >= 0.99)[0][0]
        ## lr0 does a combined left-to right sort to not do repetitive argwhere calls. T
        ##the two ends of lr0 give the start and end of above noise signal.
        lr0 = np.argwhere((np.abs(cdf) > 0.003) & (np.abs(cdf) < 0.997))

        ## remove the background noise from the profile.
        ## Take all the 4 positions denoting the IQUV measurements in one call.
        ## sobs_in subscripted by zz,xx,yy, thus the sum is over columns(axis=0) over the wavelength range.
        ## the sums give 4 values (for each iquv spectral signal) of noise before and after the line emission in the wavelength range.
        ## Finaly it divides by the amount of points summed over for each components.
        #background_1pix     = (np.sum(sobs_in_1pix[0:lr0[0,0],:],axis=0) + np.sum(sobs_in_1pix[lr0[-1,0]:,:],axis=0)) / (nw-lr0[-1,0]+lr0[0,0]+1)
        background_1pix      = np.mean(sobs_in_1pix[wcont:wcont+3:,:],axis = 0)       ## just evaluate and subtract the continuuum observed around specific wavelentghts.
        #background_1pix = 0

        ## Temporary array to store the background subtracted spectra. All four component done in one call.
        tmp2                = sobs_in_1pix[:,:] - background_1pix
        tmp2[tmp2[:,0]<0,0] = 0

        ##Now integrate/sum the four profiles
        ## tmp2 [:,0] --> Stokes I
        sobs_tot_1pix       = np.zeros((4),dtype=np.float64)
        sobs_tot_1pix[0]    = np.sum(tmp2[:,0]*dwv)                         ## background subtraction can introduce negative values unphysical for I; Sum only positive counts;

        ## tmp2 [:,3] --> Stokes V
        ## old descriptive method for estimating the position of the sign
        #svmin=np.argwhere(tmp2[:,3] == np.min(tmp2[:,3]) )[-1,0]                    ## position of minimum/negative Stokes V lobe
        #vmax=np.argwhere(tmp2[:,3] == np.max(tmp2[:,3]) )[-1,0]                    ## position of maximum/positive Stokes V lobe
        #svmin               = np.nanargmin(tmp2[:,3])                                                   ## position of minimum/negative Stokes V lobe
        #svmax               = np.nanargmax(tmp2[:,3])                                                   ## position of maximum/positive Stokes V lobe

        ## Simple sum ov Stokes V is insufficient to recover the emission coefficients because of the normalization of the profiles. See pycelp calc_stokesSpec for the formulation.
        ## We now define the factor, divide by the normalization and then median along the spectral axis while excludind edges acocunting for potential doppler shifts.
        ## The difference between the emission coefficient and this method is ~1.0045, matching pretty well.
        #sobs_tot_1pix[3]=(np.sign(svmin-svmax))*np.sum(np.abs(tmp2[:-1,3])*dwv)  ## Sum the absolute of Stokes V signal and assign sign based on svmin and svmax lobe positions
        dIdlnorm_fact       = np.gradient(tmp2[:,0]/sobs_tot_1pix[0],wlarr)                                                     ## small factor that suppresses division by 0 warnings.
        vscl                = tmp2[np.int32(0.25*nw):np.int32(0.75*nw),3] / (dIdlnorm_fact[np.int32(0.25*nw):np.int32(0.75*nw)]+1e-36)  ## selects the half innermost portion of the fit

        ## As denoted in the pycelp calc_PolEmissCoeff documentation, the Stokes V coefficient is not integrated with respect to wavelength (units with Armstrong^-1).
        ## The pycelp database has the same restriction. Thus, the SV data presented in any wavelength units, needs to be converted in Angstrom^-1 counts for matching.
        ## e.g. if the input dispersion axis is in nm, then the integrated signal needs a factor *10  to account for this calculation.
        ## We will use the feed wavelength unit to check for this (wcont is simplest). e.g. 10747 is Angstrom, 1074.7 is nm, 1.074 is mum, etc, and apply the corresponding factor.
        sobs_tot_1pix[3]    = np.median(vscl[np.isfinite(vscl)]) #* 10 ** np.rint(np.log10(10000/(wlarr[0])))
        #sobs_tot_1pix[3]    = np.mean(np.sign(svmin-svmax)*vscl[np.isfinite(vscl)])

        ## tmp2 [:,1:3] --> Stokes Q and U
        #sobs_tot_1pix[1:3]=np.sum(tmp2[:-1,1:3]*dwv,axis=0) ## sum both components in one call. These can take negative values, no need for precautions like for Stokes I.
        sobs_tot_1pix[1]    = np.sum(tmp2[:,1]*dwv)
        sobs_tot_1pix[2]    = np.sum(tmp2[:,2]*dwv)

        ## Check for finite and defined integrated totals to avoid overflows downstream. Set integrals to 0 if any of the the line sums are infinite or beyond float 64 precision
        # for kk in range(4):
        #     sobs_tot_1pix[kk] = sobs_tot_1pix[kk] if (np.isfinite(sobs_tot_1pix[kk]) and 1e-150 <= abs(sobs_tot_1pix[kk]) <= 1e150) else 1e-38

        ## Compute the rms for each quantity (phil computation). Ideally the background should be the same in all 4 measurements corresponding to one line.
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

        ## One line rms calculation based on the above formulation
        #rms_1pix = np.sqrt((background_1pix[0]/sobs_tot_1pix[:]**2 + background_1pix[0]/sobs_tot_1pix[0]**2)*((sobs_tot_1pix[:]/sobs_tot_1pix[0])**2))

        ## distribution standard deviation
        # for kk in range(4):
        #     rms[xx,yy,(4*zz)+kk] = np.std(sobs_in[zz][xx,yy,:,kk]/np.max(np.abs(sobs_in[zz][xx,yy,:,kk]))) ##a standard deviation in normalized units as normalized data is matched in cledb_matchiquv
            #rms[xx,yy,(4*zz)+kk] = (0.5*np.std(sobs_in[zz][xx,yy,0:lr0[0,0]+1,kk]/np.max(np.abs(sobs_in[zz][xx,yy,0:lr0[0,0]+1,kk])))+0.5*np.std(sobs_in[zz][xx,yy,lr0[-1,0]:,kk]/np.max(np.abs(sobs_in[zz][xx,yy,lr0[-1,0]:,kk]))) )
    # else:
        ## [issuemask].....
        ## Compute the RMS for the noise and the signal respectively then compute the SNR
        rms_noise  = np.sqrt(np.mean(np.concatenate((sobs_in_1pix[0:lr0[0,0]+1,:],sobs_in_1pix[lr0[-1,0]:,:]))**2))
        #print(lr0[0,0]+np.rint((lr0[-1,0]-lr0[0,0]/3)),lr0[0,0]+2*np.rint((lr0[-1,0]-lr0[0,0]/3)))
        #print(np.sqrt(np.mean(sobs_in_1pix[np.round(lr0[0,0]+lr0.shape[0]/3).astype(np.int16):np.round(lr0[0,0]+lr0.shape[0]*2/3).astype(np.int16) ,:]**2,axis=0)))
        rms_signal = np.sqrt(np.mean(sobs_in_1pix[np.round(lr0[0,0]+lr0.shape[0]/3).astype(np.int16):np.round(lr0[0,0]+lr0.shape[0]*2/3).astype(np.int16) ,:]**2,axis=0))  ##round to nearest integers to take the rms of the tip of the gaussian
        snr_1pix   = rms_signal / rms_noise

        ## Apply issue code
        iss_int = 0
        for kk in range(3):
            if snr_1pix[kk] < 1: iss_int = 1     ## update to 1 for any of the Stokes IQU SNR < 1
        if snr_1pix[3] <1: iss_int      += 2     ## +2  for Stokes V SNR < 1
    else:
        return xx,yy,zz,np.zeros((4),dtype=np.float32),np.zeros((4),dtype=np.float32),np.zeros((4),dtype=np.float32),0

    return xx,yy,zz,sobs_tot_1pix,snr_1pix,background_1pix,iss_int
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_qurotate(sobs_tot,keyvals,hpxygrid):
    """
    Function for rotating the linear polarization components.

    The corresponding angle are using eq. 9 & 10 from Paraschiv & Judge 2022. This implicitly assumes that yobs is kept in the same units as the header keys, either R_sun or arcsec.
    """

    ######################################################################
    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny         = keyvals[0:2]
    crpix1,crpix2 = keyvals[5:7]
    crval1,crval2 = keyvals[8:10]
    cdelt1,cdelt2 = keyvals[11:13]
    linpolref     = keyvals[14]

    if crpix1 != 0:
        crval1 = crval1 - (crpix1 * cdelt1)
    if crpix2 != 0:
        crval2 = crval2 - (crpix2 * cdelt2)

    aobs        = np.empty((nx,ny),dtype=np.float32)
    sobs_totrot = np.copy(sobs_tot)                      ## copy the sobs array as only QU need to be updated

    if np.sum(hpxygrid) == 0.:    ## rectangular grid array
        for xx in range(nx):
            for yy in range(ny):
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

    else:    ##  non-rectangular array dimensions possible for Cryo-nirsp
        for xx in range(nx):
            for yy in range(ny):
                aobs[xx,yy]= - np.arctan2(hpxygrid[0,xx,yy],hpxygrid[1,xx,yy]) ## assuming the standard convention for arcsecond coordinates <--> trigonometric circle
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
    """
    Helper function that computes the cdf distribution for a spectra.
    """
    cdf=np.zeros((spectr.shape[0]),dtype=np.float32)
    for i in range (0,spectr.shape[0]):
        cdf[i]=np.sum(spectr[0:i+1])        ## need to check if should start from 0 or not.
    return cdf[:]/cdf[-1]                      ## norm the cdf to simplify interpretation
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_dens(sobs_totrot,yobs,keyvals,params):
    """
    Compute the density using two stokes I observations (serialized parallel runs for 1 pixel at a time) of Fe XIII.

    This function is a stripdown of a standalone density calculation implementation found at https://github.com/arparaschiv/FeXIII-coronal-density/
    The function requires a CHIANTI calculation lookup table, usually placed in CLEDB's main directory.
    The lookup table can be created using a script from the above github repo.
    """

    ######################################################################
    ## Unpack the required keywords (these are unpacked so its clear what variables are being used. One can just use keyvals[x] inline.)
    nx,ny = keyvals[0:2]

    dobs           = np.empty((nx,ny),dtype=np.float32)
    iss_dens = np.zeros((nx,ny),dtype=np.int32)   ### for tracking the lookup table height applicability (code 4)
    ## theoretical chianti ratio calculations created via PyCELP or SSWIDL
    ## read the chianti table
    if (params.lookuptb[-4:] == ".npz"):                         ## default
        chianti_table        =   dict(np.load(params.lookuptb))  ## variables (h,den,rat) directly readable by work_1pix .files required for loading the data directly.
    else:
        if params.verbose >= 1: print("OBS_DENS: FATAL! CHIANTI look-up table not found. Is the path correct?")
        return dobs,iss_dens                          ## a zero array at this point
    #print(chianti_table.keys())                      ## Debug - check the arrays

    ##  in the look-up table:
    ## 'h' is an array of heights in solar radii ranged 1.01 - 2.00(pycelp)
    ## 'den' is an array of density (ranged 6.00 to 12.00 (pycelp) or 5.0 to 13.0 (sswidl); the broader interval is not really recommended here.
    ## 'rat' is a 2D array containing line ratios corresponding to the density range values at each distinct height.

    ##  To query the look-up table:
    ##  print(chia['h'],chia['h'].shape,chia['rat'].shape,chia['den'].shape)

    ## look-up table resolutions
    ## pycelp: h is of shape [99]; den and rat are arrays of shape [99, 120] corresponding to the 99 h height values and 120 density values

    ######################################################################
    ## set up the cpu worker and argument arrays and send the parallel tasks to execute
    p         = multiprocessing.Pool(processes=min(params.ncpu, multiprocessing.cpu_count()-2),maxtasksperchild = 50000)     ## dynamically defined from system query as total CPU core number - 2
    ## argument index keeper for splitting tasks to cpu cores

    arg_array = []
                                                                                              ## Raster slit branch
    for xx in range(sobs_totrot.shape[0]):
        for yy in range(sobs_totrot.shape[1]):
            arg_array.append((xx,yy,sobs_totrot[xx,yy,0],sobs_totrot[xx,yy,4],chianti_table,yobs[xx,yy]))  ## 1074/1079 ratio

    if params.verbose >= 1:       ## Some progressbar print output
        print("OBS_DENS: Calculating observation LOS plasma density:")
        rs = p.starmap(obs_dens_work_1pix,tqdm(arg_array,total=len(arg_array)))

    else:                  ## No progressbar print output\
        rs = p.starmap(obs_dens_work_1pix,arg_array)

    p.close()

    for i,res in enumerate(rs):
        xx,yy,dobs[xx,yy],iss_dens[xx,yy] = res
    dobs[dobs <= 0] = 1

    return np.round(np.log10(dobs),3),iss_dens
###########################################################################
###########################################################################

###########################################################################
###########################################################################
@njit(parallel=params.jitparallel,cache=params.jitcache)
def obs_dens_work_1pix(xx,yy,a_obs,b_obs,chianti_table,y_obs):
    """
    Work helper function for parallelizing *obs_dens* tasks.

    Compute the ratio of the observation using the 1074/1079 fraction, then interpolate to the rat component of the chianti_table.
    Finally  recover the encoded density leading up to the ratio at a specific height.
    """

    ## Sanity checks at pixel level
    if b_obs == 0:                                      ## don't divide by 0
        return xx,yy,1,0                                ## return 1 value <--np.log10(1) will give 0 in obs_dens

    ######################################################################
    ## line ratio calculations for peak line emission quantities
    rat_obs       = a_obs/b_obs        ## requires only one input number for each line
    #rat_obs_noise = rat_obs*np.sqrt((np.sqrt(a_obs)/a_obs)**2+(np.sqrt(b_obs)/b_obs)**2)    ## error propagation

    ## another sanity check for numerical errors
    if (np.isnan(rat_obs) or np.isinf(rat_obs)):        ## discard nans and infs ratios in rat_obs argument.
        return xx,yy,0,0                                ## return 0 value

    else:                                               ## main loop for valid "rat_obs" value
        ## find the corresponding height (in solar radii) for each pixel
        subh = np.argwhere(chianti_table['h'] > y_obs)

        ## if height is greater than maximum h (2.0R_sun as in the currently implemented table) just use the 2.0R_sun corresponding ratios.
        iss_d=0 ## setup an issue tracker for "code 4"
        if len(subh) == 0:
            subh = [-1]
            iss_d = 4

        ## make the interpolation function; Quadratic as radial density drop is usually not linear
        ifunc = scipy.interpolate.interp1d(chianti_table['rat'][subh[0],:].flatten(),chianti_table['den'], kind="quadratic",fill_value="extrapolate")

        ## apply the interpolation to the data
        dens_1pix       = ifunc(rat_obs)

        #dens_1pix_noise = ifunc(rat_obs+rat_obs_noise) - dens_1pix

        ## debug prints
        #print("Radius from limb: ",rpos," at pixel positions (",xx,yy,")")         ## debug
        #print(len(subh),rpos/rsun)                                                 ## debug
        return xx,yy,dens_1pix,iss_d
###########################################################################
###########################################################################

###########################################################################
###########################################################################
def iss_print(issuemask,tline=['fe-xiii_1074', 'fe-xiii_1079'],xx=0,yy=0,iss=0,map1=False):
###########################################################################
###########################################################################
    """
    Function that can be applied to the issuemask array to show punctual or map issues that were flagged during processing with two modes of running.

    a. Plot  all issues encountered in a specific pixel (default output).

    Parameters:
    ----------
    issuemask : intarr
              The encoded issue mask array.
    tline     : strarr
              The line string information from the keywords (keyvals). Default is Fe XIII 1074 and 1079.
    xx and yy : ints
              Pixel coordinates to be checked.

    Returns
    -------
    iss_text  : string
              All codes that got flagged for location xx and yy.

    b. Plot where a specific issue occurs in the entire observation map for either line.

    Parameters:
    ----------
    issuemask : intarr
              The encoded issue mask array.
    iss       : int
              Which issue to check for. Consult *sobs_preprocess*, *sdb_preprocess*, or documentation at https://cledb.readthedocs.io/en/latest/outputs.html#tentative-issuemask-implementation for codes.
    map1      : boolean
              Flag to switch to map output. This needs to be set to false for one-pixel version of this function.

    Returns
    -------
    iss_inst  : intarr
              A map of the physical shape of issuemask that shows where the requested iss issues manifests.
    """

    ######################################################################
    if map1 == False and iss == 0:      ## one pixel branch
        for ii in range(issuemask.shape[2]):
            print(f"ISS_PRINT: One pixel warnings for line {tline[ii]} at location x = {xx} and y ={yy} follow:")
            if issuemask[xx,yy,ii] == 0:
                print("No warnings recorded for this pixel")
            else:
                iss1=iss_block(issuemask[xx,yy,ii])
                for i in range(len(iss1)):
                    print(f"Code {iss1[i]}: {iss_text(iss1[i])}")
        return None ## only print, nothing to return

    elif map1 and iss != 0:           ## map branch for one issue at a time
        print(f"ISS_PRINT: A map describing locations with warning code {iss} will be returned.")
        print(f"Displaying Code {iss}: {iss_text(iss)}")
        if (issuemask == 0).all():
            print(f"No warnings of any kind recorded in the entire map.")
            return np.zeros_like(issuemask[:,:,0])
        else:
            # iss_inst=np.zeros_like(issuemask[:,:,0])
            # for xx in range(issuemask.shape[0]):
            #     for yy in range(issuemask.shape[1]):
            #         if (iss in iss_block(issuemask[xx,yy,0])) or (iss in iss_block(issuemask[xx,yy,1])):
            #             iss_inst[xx,yy] = iss
            # return iss_inst ## returns the map instance where the selected issue occured.
            iss_inst=np.zeros_like(issuemask[:,:,0])
            for xx in range(issuemask.shape[0]):
                for yy in range(issuemask.shape[1]):
                    for zz in range(issuemask.shape[2]):
                        if (iss in iss_block(issuemask[xx,yy,zz])):
                            iss_inst[xx,yy] = iss
            return iss_inst ## returns the map instance where the selected issue occured.

    else:
        print("ISS_PRINT: FATAL! Input keyword combination not understood.")
        return -1

###########################################################################
###########################################################################
def iss_block(x):
    """
    Helper function used to decode the information in the issuemask into text for *iss_print"
    iss_block is duplicated in both prepinv and procinv modules to avoid cross-loading the modules.
    """
    ##     Converting a number into its binary equivalent.
    #print ("Blocks for %d : " %x, end="")
    v = []
    ii=0
    while (x > 0):
        v.append(2**(ii*np.int32(x % 2)) if np.int32(x % 2) != 0 else 0)
        x = np.int32(x / 2)
        ii+=1
    return v
###########################################################################
###########################################################################

###########################################################################
###########################################################################
def iss_text(x):
    """
    Helper function that matches one input warning code to a human readable explanation text.
    """
    ## series of if-elif statements to avoid numba incompatibility.
    if x == 1:
        return "WARNING: One or more of Stokes I, Q, U are lower than statistical noise SNR threshold."
    if x == 2:
        return "WARNING: Stokes V is lower than statistical noise SNR threshold."
    if x == 4:
        return "WARNING: Observed height is greater than the maximum height extent of the density look-up table."
    if x == 8 or x == 16:
        return "Code reserved but not currently implemented."
    if x == 32:
        return "WARNING: Phi_B possibly influenced by the Stokes QU noise threshold or arctangent asymptote (for 1-line observations)."
    if x == 64:
        return "WARNING: Linear polarization azimuth might be close to Van-Vleck ambiguity (for 1-line observations)."
    if x == 128:
        return "WARNING: Initial Gaussian fit guess parameters not accurately found and/or signal is not following a Gaussian distribution (1-line observations)."
    if x == 256:
        return "Code reserved but not currently implemented."
    if x == 512:
        return "WARNING: Linear polarization azimuth is close to Van-Vleck ambiguity (for 2-line observations)."
    if x == 1024:
        return "WARNING: Database fit failed to converge reliably. (for 2-line obs)."
    if x == 2048:
        return "WARNING: One or more of B components, are lower than noise threshold (for 2-line observations)."
    if x == 4096:
        return "Code reserved but not currently implemented."

    return "Code not understood or implemented"
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