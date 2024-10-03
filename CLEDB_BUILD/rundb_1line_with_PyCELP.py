#!/usr/bin/env python
# coding: utf-8

# # **CLEDB PyCELP database generator**
# 
# This function helps create a forward-modelled database of Fe XIII 1074.7nm and 1079.8nm lines at various heights in the offlimb corona. Si IX, Si X, and Mg VIII atoms can be computed, but currently we do not recommend using databases of these ions. More work is needed to understand them.
# 
# **Notes**
# 
# - If using these scripts, please consider referencing the [CLEDB](https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract), [PyCELP](https://ui.adsabs.harvard.edu/abs/2020SoPh..295...98S/abstract), and [CHIANTI](https://www.chiantidatabase.org/referencing.html) publications.
# 
# 
# - More information on using the PyCELP Ion class can be found in a [dedicated example notebook](https://github.com/tschad/pycelp/blob/main/examples/A_tour_of_the_Ion_class.ipynb).
# 
# - Last updated: 2024 June 28 -- Calculations currently use the [CHIANTI 10.1](https://www.chiantidatabase.org/chianti_download.html) database and [PyCELP](https://github.com/tschad/pycelp) commit 7f72443.
# 
# 
# Contact: Alin Paraschiv, NSO ---- arparaschiv at nso edu
# 
# <div class="alert alert-warning" role="alert">
#     <div class="row vertical-align">
#         <div class="col-xs-1 text-center">
#             <i class="fa fa-exclamation-triangle fa-2x"></i>
#         </div>
#         <div class="col-xs-11">
#                 <strong>Attention:</strong> these calculations are intensive. Database generation is generally only needed once.
#         </div>   
#     </div> 
# </div>

# In[3]:


import pycelp
import numpy as np
#import matplotlib.pyplot as plt
#plt.rcParams['figure.dpi'] = 150
#%matplotlib widget
import os,sys,shutil
import pickle
import multiprocessing
from tqdm import tqdm
os.environ["XUVTOP"] = './config/pycelp/CHIANTI_10.1_database/' ## If you havent already set the environment variable XUVTOP for the location of the database, set it here


# In[ ]:


def gen_pycelp_db(na = 0, mpc = multiprocessing.cpu_count()-1):
    ## Constants and useful variables with respect to this specific calculation
    linestr = [["fe-xiii_1074/","fe-xiii_1079/","si-x_1430/","si-ix_3934/","mg-viii_3028/"],
              ["fe_13",         "fe_13",        "si_10",     "si_9",       "mg_8"       ],
              ["10746",         "10798",        "14300",     "39281",      "30276"]]

    print("############# CLEDB BUILD ######################################")
    print("The database inversion methodology is ready for Fe XIII line pairs.")
    print("Other lines, even if available are not yet ready for production runs")
    print("################################################################")
    if (na < 1) or (na > 5):
       return "Line option incorrect, or no line given! Aborting"
    else:
       print("We are building a database of " + linestr[0][na-1][:-6] + " at " + linestr[2][na-1][:-1] + " nm.")

    la                       = 25                   ## Number of levels to include when doing the statistical equiblibrium and radiative transfer equations. The database has ~750 levels. Anything above 80 levels should be enough. default 25 levels is underoptimal!
    sel_ion                  = pycelp.Ion(linestr[1][na-1],nlevels = la)
    electron_temperature     = sel_ion.get_maxtemp() ## kelvins; maximum formation temperature for selected ion. We will do all calculations under this assumption.This implies we are approximating density only corresponding to plasma around this temperature.
    magnetic_field_amplitude = 1.0                   ## Calculate the database for 1G fields, then just scale linearly with Stokes V amplitude in CLEDB

    ## create the folders for the database files
    ## ALL EXISTING DATABASES OF THE SELECTED LINES WILL BE DELETED
    if os.path.exists(linestr[0][na-1][:-1]):
        shutil.rmtree(linestr[0][na-1][:-1])
        os.makedirs(linestr[0][na-1][:-1])
    else:
        os.makedirs(linestr[0][na-1][:-1])

    ## The grid space defined in DB.INPUT allows sampling 10 density values per order of density magnitude with line ratios sensible to significant measurements of two decimals.## Read the database dimensions in all 5 axes.
    ## The density and height samplings and spacings are fixed to the values given by the chianti look-up table. DO NOT CHANGE WITHOUT ALSO REGENERATING THE LOOKUP TABLE.
    f1 = np.loadtxt("./DB.INPUT",dtype=np.int32, comments='*',max_rows=1)
    f2 = np.loadtxt("./DB.INPUT",dtype=np.float32, comments='*',skiprows=6)

    heights    = np.round(np.linspace(f2[2],f2[3],np.int32(f1[0]),dtype=np.float32),2) ## solar radius units. This corresponds to offlimb heights of 1-2 solar radius, with intervals of 0.01
    densities  = np.round(10.**np.linspace(f2[0],f2[1],f1[1],dtype=np.float32),3)      ## array of densities to be probed. This gives roughly 10 density samples for each order of magnitude in the logarithmic density space spanning 10^6-10^12 electrons.
    losdepth   = np.linspace(f2[4],f2[5], f1[2],dtype=np.float32)                      ## los depth to be sampled; 1 radius centered in the POS, with 0.05 intervals
    ## phic and thetac are defined to not double up the angles = 2pi and respectively angles = pi as they are the same Stokes coefficients as at angle=0. Create the intervals such that the number od discreet points and interval does not include the last point in the range. In the default settings this creates 180 phi ponts spread in the range 0-178deg with 2 degree res.
    phic       = np.linspace(f2[6],f2[7], f1[3]+1,dtype=np.float32)[:-1]               ## Phi_b POS azimuthal angle; rad to deg; 2 deg resolution
    thetac     = np.linspace(f2[8],f2[9], f1[4]+1,dtype=np.float32)[:-1]               ## Theta_b LOS longitudinal angle;rad to deg; 2 deg resolution

    ## write a header file that stays with the generated database.
    with open( linestr[0][na-1] + 'db.hdr', 'w' ) as f:
        f.write( '  ' + str(len(heights)) + '  ' + str(len(densities)) + '  ' + str(len(losdepth)) + '  ' + str(len(phic)) + '  ' +str(len(thetac)) + '\n' )
        f.write( '  ' + str(np.round(np.log10(densities[0]),2)) + '  ' +  str(np.round(np.log10(densities[-1]),2))  + '\n' )
        f.write( '  ' + str(losdepth[0]) +'  '+ str(losdepth[-1]) + '\n' )
        f.write( '  ' + str(np.round(phic[0],8))   + '  ' + str(np.round(phic[-1],8)) + '\n' )
        f.write( '  ' + str(np.round(thetac[0],8)) + '  ' + str(np.round(thetac[-1],8)) + '\n' )
        f.write( '  ' + "1" + '\n' )
        f.write( '  ' + str(np.round(sel_ion.get_emissionLine((np.int32(linestr[2][na-1]))).wavelength_in_air,8)) + '\n' )
        f.write( '  ' + "1")      ## this denotes a pycelp database

    ## 5 level loop to compute the database entries
    ## loop has the order height --> density --> losdepth --> phi--> theta for PyCELP efficiency

    ## set up the cpu worker and argument arrays
    if mpc > multiprocessing.cpu_count()-4:
        mpc   = multiprocessing.cpu_count()-4
        print("More CPU threads than available are requested! Using system max - 1 threads.")

    p         = multiprocessing.Pool(processes = mpc, maxtasksperchild = 10) ## dynamically defined from system query as total CPU core number - 1
    ## argument index keeper for splitting tasks to cpu cores
    arg_array = []

    for h,eheight in enumerate(heights):                              ## apparent (projected) height; same sampling as the CHIANTI density lookup table
        for d,edens in enumerate(densities):                          ## sampled densities; same sampling as the CHIANTI density lookup table
            arg_array.append((eheight,edens,losdepth,phic,thetac,electron_temperature,magnetic_field_amplitude,la,linestr,na)) ## Only one header instance goes in for oa set of maps. pointing should be the same in both.

    rs        = p.starmap(work_heighttimesdens,arg_array)
    p.close()
    # pbar      = tqdm(total=len(arg_array))  ## progress bar implmentation via tqdm
    # for i,res in enumerate(rs):
    #     pbar.update()
    # pbar.close()

    return "Database calculations completed."


def work_heighttimesdens(eheight,edens,losdepth,phic,thetac,electron_temperature,magnetic_field_amplitude,la,linestr,na):

    iquv_database = np.zeros((len(losdepth), len(phic), len(thetac), 4),dtype=np.float32) ## arrays to store 1 line of data; height x dens individual files will be written
    sel_ion_a     = pycelp.Ion(linestr[1][na-1],nlevels = la)
    for l,elosdepth in enumerate(losdepth):                                               ## sampling for line of sight depth of main emitting structure

        ## database encoded parameters
        eheighteff = np.sqrt(eheight**2 + elosdepth**2)                              ## allignment is a function for local radial height computed here.
        evert = 0                                                                         ## restricted to the z=0 plane; IV are invariant in Z, QU can be rotated for any other z planes
        alpha = - np.arctan2(elosdepth,eheight)                                           ## Theoretically, this is np.arctan2(gx, np.sqrt(gy**2+gz**2)) in 3D space. -90--90; 0 in POS
        beta  =   np.pi/2 #np.arctan2(eheight,evert)                                      ## In the z=0 case, this is always pi/2
        theta =   0.5 * np.pi + alpha                                                     ## Scattering angle corresponding to a vertical reference direction for linear polarization (r.d.l.p) when gamma=0

        for p,ephic in enumerate(phic):                                                   ## cartesian azimuth angle
            for t,ethetac in enumerate(thetac):                                           ## cartesian LOS angle

                ## Unit magnetic field components from angles spawned by database in the sun center cartezian frame Z vertical, Y along POS, X along LOS
                xb = np.sin(ethetac) * np.cos(ephic)
                yb = np.sin(ethetac) * np.sin(ephic)
                zb = np.cos(ethetac)

                ## rotations of projections around x-axis
                yyb =   np.cos(beta)  * yb - np.sin(beta)  * zb
                yzb =   np.sin(beta)  * yb + np.cos(beta)  * zb

                xxb =   np.cos(alpha) * xb + np.sin(alpha) * yzb
                zzb = - np.sin(alpha) * xb + np.cos(alpha) * yzb

                ## Compute the magnetic angles along the LVS (labeled s-frame in some works)
                ## These are varthetab and varphib in eq 42 44 of CJ99 and sketched in figure 5, (angles between LVS and B in the planes containing B LVS and k LVS
                ethetab = np.arctan2(np.sqrt(xxb**2 + yyb**2),zzb)
                ephib   = np.arctan2(yyb,xxb)

                ## LOS projected magnetic angles capital Theta_B and Gamma_B
                ethetalosb = np.arccos(  np.cos(theta) * np.cos(ethetab) + np.sin(theta) * np.sin(ethetab) * np.cos(ephib) )                      ## CJ99 eq42
                egammalosb = np.arccos(( np.sin(theta) * np.cos(ethetab) - np.cos(theta) * np.sin(ethetab) * np.cos(ephib)) / np.sin(ethetalosb)) ## CJ99 eq44; gammalosb=pi-phiblos; this is going into the geometric tensors.

                ## Calculate the statistical equilibrium
                sel_ion_a.calc_rho_sym(edens, electron_temperature, np.round(eheighteff-1.00,2), ethetab*180/np.pi, include_limbdark=True, include_protons=True)

                ## Finally compute the emission coefficients corresponding to calc_rho_sym calculation of the selected line and write the Stokes profiles in the database
                iquv_database[l,p,t,:] = sel_ion_a.get_emissionLine(np.int32(linestr[2][na-1])).calc_PolEmissCoeff( magnetic_field_amplitude, thetaBLOSdeg=ethetalosb*180/np.pi, azimuthBLOSdeg=egammalosb*180/np.pi)
    np.save('./'+linestr[0][na-1]+'DB_h'+"{:d}".format(np.int32(np.round(eheight*100)))+'_d'+"{:d}".format(np.int32(np.round(np.log10(edens)*100,2)))+'.npy',iquv_database)  ## save a losdepth x thetab x phib database file
    #return 1


## a loop format that is more useful for a pycelp ingestion if calculating ThetaBLOS and PhiBLOS can be computed cin a clever way --- WIP
## 5 level loop to compute the database entries
## loop has the order height --> density --> losdepth --> thetab --> phib for PyCELP eff()iciency
## The database array switchex thetab with phib, to preserve the same format as the older CLE database 

# for h,eheight in enumerate(tqdm(height)):                                    ## apparent (projected) height; same sampling as the CHIANTI density lookup table
#     for d,edens in enumerate(densities):                                     ## sampled densities; same sampling as the CHIANTI density lookup table     
#         iquv_database = np.zeros((len(losdepth), len(phib), len(thetab), 4)) ## arrays to store 1 line of data; height x dens individual files will be written
#         for l,elosdepth in enumerate(losdepth):                              ## sampling for line of sight depth of main emitting structure
#             eheighteff = np.sqrt(eheight**2 + elosdepth**2)                  ## allignment is a function for local radial height computed here.
#             for t,ethetab in enumerate(thetab):                              ## magnetic LOS angle
#                 sel_ion.calc_rho_sym(edens, electron_temperature, eheighteff, ethetab, include_limbdark=True, include_protons=True)
#                 ln = sel_ion.get_emissionLine(linestr[2][na-1])                ## generate the emission line parameters corresponding to calc_rho_sym calculation
#                 ethetablos=?????                                             ## Convert thetab angle from LVS to LOS 
#                 for p,ephib in enumerate(phib):                              ## magnetic azimuth angle 
#                     iquv_database[l,p,t,:] = ln.calc_PolEmissCoeff( magnetic_field_amplitude, thetaBLOSdeg=ethetablos, azimuthBLOSdeg=ephib)
#         np.save('./'+linestr[0][na-1]+'DB_h'+str(eheight)+'_d'+str(edens)+'.npy',iquv_database)  ## save a losdepth x thetab x phib database file       


# In[ ]:


if len(sys.argv) == 2:             ## script is appended, so 1 parameter is length 2
    na  = np.int32(sys.argv[1])  
    gen_pycelp_db(na)
elif len(sys.argv) == 3:           ## script is appended, so 2 parameters is length 3
    na  = np.int32(sys.argv[1])  
    mpc = np.int32(sys.argv[2])
    gen_pycelp_db(na,mpc)
elif len(sys.argv) > 2:
    print("Too many input arguments")
else:    
    print("No arguments given. Database needs a line selection!")


# In[ ]:





# In[ ]:


### debug codes


# In[ ]:


# def work_heighttimesdens(eheight,edens,losdepth,phic,thetac,electron_temperature,magnetic_field_amplitude,sel_ion):

#     for l,elosdepth in enumerate(losdepth):                                               ## sampling for line of sight depth of main emitting structure

#         ## database encoded parameters
#         eheighteff = np.sqrt(eheight**2 + elosdepth**2)                                   ## allignment is a function for local radial height computed here.
#         evert = 0                                                                         ## restricted to the z=0 plane; IV are invariant in Z, QU can be rotated for any other z planes
#         alpha = - np.arctan2(elosdepth,eheight)                                           ## Theoretically, this is np.arctan2(gx, np.sqrt(gy**2+gz**2)) in 3D space. -90--90; 0 in POS
#         beta  =   np.arctan2(eheight,evert)                                               ## In the z=0 case, this is always pi/2
#         theta =   0.5 * np.pi + alpha                                                     ## Scattering angle corresponding to a vertical reference direction for linear polarization (r.d.l.p) when gamma=0

#         for p,ephic in enumerate(phic):                                                   ## cartesian azimuth angle
#             for t,ethetac in enumerate(thetac):                                           ## cartesian LOS angle

#                 ## Unit magnetic field components from angles spawned by database in the sun center cartezian frame Z vertical, Y along POS, X along LOS
#                 xb = np.sin(ethetac) * np.cos(ephic)
#                 yb = np.sin(ethetac) * np.sin(ephic)
#                 zb = np.cos(ethetac)

#                 ## rotations of projections around x-axis
#                 yyb =   np.cos(beta)  * yb - np.sin(beta)  * zb
#                 yzb =   np.sin(beta)  * yb + np.cos(beta)  * zb

#                 xxb =   np.cos(alpha) * xb + np.sin(alpha) * yzb
#                 zzb = - np.sin(alpha) * xb + np.cos(alpha) * yzb

#                 ## Compute the magnetic angles along the LVS (labeled s-frame in some works)
#                 ## These are varthetab and varphib in eq 42 44 of CJ99 and sketched in figure 5, (angles between LVS and B in the planes containing B LVS and k LVS
#                 ethetab = np.arctan2(np.sqrt(xxb**2 + yyb**2),zzb)
#                 ephib   = np.arctan2(yyb,xxb)

#                 ## LOS projected magnetic angles capital Theta_B and Gamma_B
#                 ethetalosb = np.arccos(  np.cos(theta) * np.cos(ethetab) + np.sin(theta) * np.sin(ethetab) * np.cos(ephib) )                      ## CJ99 eq42
#                 egammalosb = np.arccos(( np.sin(theta) * np.cos(ethetab) - np.cos(theta) * np.sin(ethetab) * np.cos(ephib)) / np.sin(ethetalosb)) ## CJ99 eq44; gammalosb=pi-phiblos; this is going into the geometric tensors.                   

#                 #print (ethetalosb,egammalosb)
#                 ## Calculate the statistical equilibrium
#                 sel_ion.calc_rho_sym(edens, electron_temperature, eheighteff, ethetab*180/np.pi, include_limbdark=True, include_protons=True)

#                 ## Finally compute the emission coefficients corresponding to calc_rho_sym calculation of the selected line and write the Stokes profiles in the database
#                 aa= sel_ion.get_emissionLine(10747).calc_PolEmissCoeff( magnetic_field_amplitude, thetaBLOSdeg=ethetalosb*180/np.pi, azimuthBLOSdeg=egammalosb*180/np.pi) 
#                 print(aa)

#     return 1


# In[ ]:


# la                       = 50                    ## Number of levels to include when doing the statistical equiblibrium and radiative transfer equations. The database has ~750 levels. Anything above 300 levels should be enough.
# sel_ion                  = pycelp.Ion("fe_13",nlevels = la)
# electron_temperature     = sel_ion.get_maxtemp() ## kelvins; maximum formation temperature for selected ion. We will do all calculations under this assumption.This implies we are approximating density only corresponding to plasma around this temperature.
# magnetic_field_amplitude = 1.0                   ## Calculate the database for 1G fields, then just scale linearly with Stokes V amplitude in CLEDB


# f1 = np.loadtxt("./DB.INPUT",dtype=np.int32, comments='*',max_rows=1)
# f2 = np.loadtxt("./DB.INPUT",dtype=np.float32, comments='*',skiprows=6)

# heights    = np.round(np.linspace(f2[2],f2[3],np.int32(f1[0]),dtype=np.float32),2) ## solar radius units. This corresponds to offlimb heights of 1-2 solar radius, with intervals of 0.01
# densities  = np.round(10.**np.linspace(f2[0],f2[1],f1[1],dtype=np.float32),3)      ## array of densities to be probed. This gives roughly 10 density samples for each order of magnitude in the logarithmic density space spanning 10^6-10^12 electrons. 
# losdepth   = np.linspace(f2[4],f2[5], f1[2],dtype=np.float32)                      ## los depth to be sampled; 1 radius centered in the POS, with 0.05 intervals
# phic       = np.linspace(f2[6],f2[7], f1[3],dtype=np.float32)#*180/np.pi            ## Phi_b POS azimuthal angle; rad to deg; 2 deg resolution
# thetac     = np.linspace(f2[8],f2[9], f1[4],dtype=np.float32)#*180/np.pi            ## Theta_b LOS longitudinal angle;rad to deg; 2 deg resolution



# work_heighttimesdens(1.1,1e8,losdepth[:1],phic,thetac,electron_temperature,magnetic_field_amplitude,sel_ion)

