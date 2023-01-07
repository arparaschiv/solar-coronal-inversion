### **CLEDB coronal inversion UPDATES and TODO list**

**Github update history**

| Commit Tag | Date | Description |
|:---------|:-----:|:-----|
| *initial* | 20210802 | Self explanatory.|
| *update-slurm* | 20210828 | Implemented initial slurm-enabled and batch versions for both jupyter and database build scripts.<br>Updated and extended documentation. |
| *update-numbaproc* | 20210906 | Implemented a parallel enable/disable keyword for numba enhanced functions.<br>Functions that don't benefit from parallel splitting are hard-coded to parallel=False. |
| *update-tidbits* | 20210908 | Small updates for optional plotting; save/load datacubes, etc.<br>Implemented slurm tidbits; standard scripts can now be run inside interactive sessions on RC systems.<br>Redid how calculations are computed in the CLEDB_BLOS module.<br>Updated constants and ctrlparams classes.|
| *update-CLEDB_BUILD* | 20210915 | Updated database building scripts.<br>Fully implemented the batch run version with $scratch partition use.<br>Updated documentation accordingly.|
| *update-databugfixes* | 20220816 | Bug fixes update after running comp inversions. <br> Fixed bugs in obs_qurotate (prepinv module) and cledb_quderotate (proc module). <br> Fixed the cledb_obsderotate (proc module) function. <br> Added angle correction (azimuth derotate) to bphi calculation in cledb_phys (proc module). <br> cledb_getsubset routine (proc module) is optimized and re-written. <br> Introduced the "integrated" control parameter. <br> Updated the conda distribuition and python 3.10, numpy, numba, etc. <br> Bugfix update to CLE 2.0.3 executables in BUILD|
| *update-iqud* | 20230106 | Major update. <br> CURRENTLY INCOMPLETE. If any problems arice. DOWNLOAD previous commit tag. <br> CLE is updated to 2.0.4. A number of issues with the forward-synthesis are resolved as outlined in the CLE README. <br> Small fix to height and depth calculations. Databases need to be recomputed with CLE >=2.0.4 <br> The database compression routine *sdb_dcompress* introduced numerical instability at particular field geometries. It is now disabled. <br> Databases are now of type np.float32. Consequentially, double precision in CLEDB is removed. <br> Noise statistics and observation rms are enabled <br> Partially implemented the IQUD (IQU+Doppler) functionality to invert vector fields in the absence of Stokes V. <br> Introduced the additional *ctrlparams* "iqud" keyword that works alongside "integrated" control parameter. <br> (not ready)Modified functions in the proc module that perform matches for iqud setups. New *cledb_matchiqud* function is added. <br> Keyword ingestion partially implemented (new *obs_headstructproc* function). Prepinv can now recognize CoMP/uCoMP keywords at input. <br> (not ready) A new CoMP test example notebook and script *test_2line_iuqd* is available. <br> Numba activate/deactivate and caching enabled/disabled keywords are implemented in *ctrlparams*. <br> Updated the numba, numpy, and scipy package versions under the currently stable python 3.10 conda env. <br> (not ready) Updated the documentation to describe the iqud setup for line-integrated observations.| 

**TODO list last update:** 20230106

1.  Add the ISSUEMASK setup as outlined in the documentation.

~~2.  Needed observation Keywords are currently manually implemented, where a synthetic CLE observation is used. 
    Information from observation keywords will need to be ingested after observations will become available.~~*update-iqud* Keywords are now ingested with data. 

3.  Develop and use a public test case for more convoluted MURAM data.

~~4.  Implement additional numba compiler flags and options. Make numba active/disabled with a ctrlparam. Implement numba caching.~~*update-iqud*

~~5.  After more information on input data is obtained, implement the 
    LEV2CALIB_WAVE and LEV2CALIB_ABSINT functions as outlined in the documentation.~~*update-iqud* DKIST lev-1 data will contain the needed corrections.

6.  implement ML_LOSDISENTANGLE Disentangling multiple LOS contributions in observations, as outlined in the documentation.

~~7.  Refine the noise uncertainty (RMS computations and usage).~~*update-iqud*