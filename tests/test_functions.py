#!/usr/bin/env python
# coding: utf-8

## set of unit tests to check the accurate execution of the main functions of CLEDB

### Sample data is generated via pycelp as follows, where input parameters can be seen:
# import pycelp
# fe13 = pycelp.Ion('fe_13',nlevels = 100)
# fe13.calc_rho_sym(1.17e8,fe13.get_maxtemp(), 0.05, 0, include_limbdark=True, include_protons=True)
# ln1 = fe13.get_emissionLine(10747.)
# ln2 = fe13.get_emissionLine(10798.)
# aat = ln1.calc_PolEmissCoeff(6,thetaBLOSdeg=30,azimuthBLOSdeg=30,)
# aas = ln1.calc_stokesSpec(6,thetaBLOSdeg=30,azimuthBLOSdeg=30,doppler_velocity=16.0, non_thermal_turb_velocity=0.0, doppler_spectral_range=(-100, 100), specRes_wv_over_dwv=200000)
# bbt = ln2.calc_PolEmissCoeff(6,thetaBLOSdeg=30,azimuthBLOSdeg=30,)
# bbs = ln2.calc_stokesSpec(6,thetaBLOSdeg=30,azimuthBLOSdeg=30,doppler_velocity=16.0, non_thermal_turb_velocity=0.0, doppler_spectral_range=(-100, 100), specRes_wv_over_dwv=200000)
# np.savez(f'../tests/sample_test_stokesspectra',aat=aat,bbt=bbt,awl=aas[0],aas=aas[1]*10,bwl=bbs[0],bbs=bbs[1]*10,allow_pickle=True)

## The Stokes V total signal is not normalized with respect wo wavelength. Units are Angstrom^-1. We thus m,ultiply the spectral outputs by 10 to present things in units of nm^-1 as in the below head_in header.

import pytest
import numpy as np
#import sys
#sys.path.append("../")      ## append root folder so we can import the modules
## package functions import
import CLEDB_PROC.CLEDB_PROC as procinv
import CLEDB_PREPINV.CLEDB_PREPINV as prepinv
## constants and control params import
import constants as consts
import ctrlparams
params=ctrlparams.ctrlparams() ##Initialize and use a shorter label
params.verbose = 0 ## set the verbosity to none for the purpose of testing

## Load the test data
try:
    data = np.load("sample_test_stokesspectra.npz")
except FileNotFoundError:
    data = np.load("./tests/sample_test_stokesspectra.npz")

sobs_in = [data["aas"].reshape(1,1,134,4),data["bbs"].reshape(1,1,134,4)]
head_in =[{'CRPIX1':0, 'CRPIX2':0, 'CRPIX3':0, 'CRVAL1': -0.30, 'CRVAL2': 1.05, 'CRVAL3': 1074.2571372, 'CDELT1': 0.0001, 'CDELT2':  0.0001, 'CDELT3': 0.00538654, 'LINEWAV': 1074.6153, 'INSTRUME': "PyCELP"},\
          {'CRPIX1':0, 'CRPIX2':0, 'CRPIX3':0, 'CRVAL1': -0.30, 'CRVAL2': 1.05, 'CRVAL3': 1079.4203968, 'CDELT1': 0.0001, 'CDELT2':  0.0001, 'CDELT3': 0.00541243, 'LINEWAV': 1079.7803, 'INSTRUME': "PyCELP"}]

## test all variables in sobs_preprocess with respect to known outputs and inside known tolerances for numerical errors.
@pytest.fixture
def fix_sobs_preprocess():
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs=prepinv.sobs_preprocess(sobs_in,head_in,params)
    return sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs

@pytest.fixture
def fix_sdb_preprocess(fix_sobs_preprocess):
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs = fix_sobs_preprocess
    params.verbose = 4 ## unique extended output for testing. not needed for normal runs
    db_enc,database,dbhdr,dbnames,db_enc_fln,db_u=prepinv.sdb_preprocess(yobs,dobs,keyvals,wlarr,params)
    params.verbose = 0
    return db_enc,database,dbhdr,dbnames,db_enc_fln,db_u

def test_sobs_preprocess(fix_sobs_preprocess):
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs = fix_sobs_preprocess

    np.testing.assert_allclose(sobs_tot, np.concatenate((data["aat"],data["bbt"]),dtype=np.float32).reshape(1,1,8),atol=1e-12)    ###tolerant to 1*StokesV levels.
    np.testing.assert_allclose(sobs_totrot, np.array([ 2.7957712e-09,  5.8281428e-11, -3.1131424e-11, -1.2028062e-13,\
                                                      9.8659880e-10,  1.8144947e-12, -9.6922470e-13, -4.1924003e-14]).reshape(1,1,8),atol=1e-14)
    np.testing.assert_allclose(snr,np.array([2.731340e+02, 3.227606e+00, 5.590377e+00, 1.009754e-01,\
                                             2.732097e+02, 2.848313e-01, 4.933422e-01, 9.935237e-02]).reshape(1,1,8),atol=1e-2)
    np.testing.assert_allclose(yobs,1.09,atol=1e-2)
    np.testing.assert_allclose(aobs,0.27,atol=1e-2)
    np.testing.assert_allclose(dobs,7.95,atol=5e-2) ## Strictly simulationwise this is 8.068(1.17e8), but due to increasing the "observed height" via CRVAL1, the ratio leads to a lower density.
    np.testing.assert_equal(keyvals[0], 1)
    np.testing.assert_equal(keyvals[1], 1)
    np.testing.assert_equal(keyvals[2], 134)
    np.testing.assert_equal(keyvals[3], 2)
    np.testing.assert_equal(keyvals[4],['fe-xiii_1074', 'fe-xiii_1079'])
    np.testing.assert_equal(keyvals[5], 0)
    np.testing.assert_equal(keyvals[6], 0)
    np.testing.assert_equal(keyvals[7], [0,0])
    np.testing.assert_equal(keyvals[8], -0.3)
    np.testing.assert_equal(keyvals[9], 1.05)
    np.testing.assert_equal(keyvals[10], [1074.2571, 1079.4204])
    np.testing.assert_equal(keyvals[11], 1e-04)
    np.testing.assert_equal(keyvals[12], 1e-04)
    np.testing.assert_equal(keyvals[13], [np.float32(0.00538654), np.float32(0.00541243)])
    np.testing.assert_equal(keyvals[14], 0)
    np.testing.assert_equal(keyvals[15], 0)

def test_sdb_preprocess(fix_sobs_preprocess,fix_sdb_preprocess):
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs = fix_sobs_preprocess
    db_enc,database,dbhdr,dbnames,db_enc_fln,db_u                              = fix_sdb_preprocess

    ## tests whether you load the correct database to match the observation
    ## To work with the github actions, we make a small mini database direc tory with 3 entries, out of which, one is the correct entry based on metadata.
    np.testing.assert_equal(db_enc, 0)     ## only one database is loaded here. Encoding is database[0].
    np.testing.assert_equal(db_u, 1)       ## unique database entry in list of available databases ## 1048 index in the full example database
    np.testing.assert_equal([dbnames[0][db_u[0]][-30:],dbnames[1][db_u[0]][-30:]],['/fe-xiii_1074/DB_h109_d795.npy', '/fe-xiii_1079/DB_h109_d795.npy']) ## this matches yobs and dobs from above in both lines


def test_spectro_proc(fix_sobs_preprocess):
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs = fix_sobs_preprocess
    specout = procinv.spectro_proc(sobs_in,sobs_tot,snr,issuemask,background,wlarr,keyvals,consts,params)

    ##check the spectroscopy outputs in terms of known output. Computed Doppler is 16km/s width is / L fraction is / P fraction is /
    np.testing.assert_allclose(specout[0,0,0,:],[1074.6726, 0.05725, 15.9715, 1.9672e-08, 2.3246e-10, -4.0264e-10, -9.0775e-12, 9.0549e-20, 0.13328, 0.057055, 0.02363386, 0.02363390],atol=1e-4) ## 1074
    np.testing.assert_allclose(specout[0,0,1,:],[1079.8379, 0.05761, 15.9968, 6.9092e-09, 7.2031e-12, -1.2476e-11, -3.1360e-12, 5.0446e-15, 0.13391, 0.057319, 0.00208507, 0.00208550],atol=1e-4) ## 1079

def test_blos_proc(fix_sobs_preprocess):
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs = fix_sobs_preprocess
    blosout=procinv.blos_proc(sobs_tot,snr,issuemask,keyvals,consts,params)

    ## simulated data with Gamma_b = 30deg (Phi_B= pi=Gamma_B; e.g. 270 or -30)  and B_los ~5Gauss
    np.testing.assert_allclose(blosout[:,:,0,3]*180/np.pi,-30, atol = 1e-2)
    np.testing.assert_allclose(blosout[:,:,1,3]*180/np.pi,-30, atol = 1e-2)

    ## Check roughly the three magnetograph interpretations. the tolerance is higher to account for differences and for the low polarization signals in 1079 increase ucnertainties.
    np.testing.assert_allclose(blosout[:,:,0,0],5.10, atol = 5e-1)
    np.testing.assert_allclose(blosout[:,:,1,0],5.10, atol = 5e-1)
    np.testing.assert_allclose(blosout[:,:,0,1],5.10, atol = 5e-1)
    np.testing.assert_allclose(blosout[:,:,1,1],5.10, atol = 5e-1)
    np.testing.assert_allclose(blosout[:,:,0,2],5.10, atol = 5e-1)
    np.testing.assert_allclose(blosout[:,:,1,2],5.10, atol = 5e-1)

def test_cledb_invproc(fix_sobs_preprocess,fix_sdb_preprocess):
    sobs_tot,yobs,snr,background,issuemask,wlarr,keyvals,sobs_totrot,aobs,dobs = fix_sobs_preprocess
    db_enc,database,dbhdr,dbnames,db_enc_fln,db_u                              = fix_sdb_preprocess

    invout,sfound=procinv.cledb_invproc(sobs_totrot,0,database,db_enc,yobs,aobs,dobs,snr,issuemask,dbhdr,keyvals,params.nsearch,params.maxchisq,0,False,params.ncpu,False,params.verbose)
    #invout,sfound=procinv.cledb_invproc(sobs_totrot,sobs_dopp,database,db_enc,yobs,aobs,dobs,snr,issuemask,dbhdr,keyvals,params.nsearch,params.maxchisq,params.bcalc,params.iqud,params.ncpu,params.reduced,params.verbose)

    ## check first solution match to the observation.
    np.testing.assert_allclose(sobs_tot[0,0,:], sfound[0,0,0,:],atol=5e-12) ## not really accurate to Stokes V, but the database accuracy in Stokes QU enforces this higher level.
    ##np.testing.assert_(np.isin(invout[0,0,1,0],[83661,83649])) ## check if first solution is in the best degenerate test. Disabled as this is a database dependent number that will change with regenerating
    np.testing.assert_equal(invout[0,0,0,2],dobs) ## check if the correct density is matched.
    np.testing.assert_equal(invout[0,0,0,3],yobs) ## check if the correct obs height is matched.
    np.testing.assert_allclose(invout[0,0,0,5],6,atol=3) ##The calculated field strength is 6 gauss. There is no fitting for stokes V, and uncertainties result from fitting the LOS and density. It will be close, but not perfect.
    np.testing.assert_equal(np.sqrt(invout[0,0,0,8]**2+invout[0,0,0,9]**2+invout[0,0,0,10]**2), invout[0,0,0,5])## check that the index matching anf the B calculation through indexes returns expected values.

    ## Now test the intercomparison between matching with IQUV or IQUD; Bpos from simulation is ~4.60
    invout2,sfound2=procinv.cledb_invproc(sobs_totrot,np.array((4.60)).reshape(1,1,1),database,db_enc,yobs,aobs,dobs,snr,issuemask,dbhdr,keyvals,params.nsearch,params.maxchisq,3,True,params.ncpu,False,params.verbose)
    np.testing.assert_allclose(sobs_tot[0,0,:], sfound2[0,0,0,:],atol=5e-12) ## Are we getting a very similar if not the same solution vector?
    np.testing.assert_(np.isin(invout[0,0,:2,0],invout2[0,0,:4,0]).all())    ## Check that both solutions of the full inversion are in the set of 4 solutions of the IQUD inversion.
    np.testing.assert_allclose(invout2[0,0,0,5],6,atol=3)                    ## Are we getting a similar field strength scaling? This is on papermore accurate than scaling minuscule Stokes V signals, but will depend on measuring waves propagation.
    np.testing.assert_allclose(np.sqrt(invout2[0,0,0,8]**2+invout2[0,0,0,9]**2+invout2[0,0,0,10]**2), invout2[0,0,0,5],atol=3)## check that the index matching anf the B calculation through indexes returns expected values in the IQUD case. Indexes ARE handled differently
    ##TBI Acheck of reduced vs full searcch