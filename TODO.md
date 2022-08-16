### **CLEDB coronal inversion TODO AND UPDATES**

**TODO list last update:** 20210802

1.  Add the ISSUEMASK setup as outlined in the documentation.

2.  Needed observation Keywords are currently manually implemented, where a synthetic CLE observation is used. 
    Information from observation keywords will need to be ingested after observations will become available.

3.  Develop and use a test case for more convoluted MURAM data.

4.  Implement additional numba compiler flags and options. Make numba active/disabled with a ctrlparam. Implement numba caching.

5.  After more information on input data is obtained, implement the 
    LEV2CALIB_WAVE and LEV2CALIB_ABSINT functions as outlined in the documentation.

6.  implement ML_LOSDISENTANGLE Disentangling multiple LOS contributions in observations, as outlined in the documentation.

7.  Refine the noise uncertainty (RMS computations and usage).

**Github update history**

| Commit Tag | Date | Description |
|:---------|:-----:|:-----|
| *initial* | 20210802 | Self explanatory.|
| *update-slurm* | 20210828 | Implemented initial slurm-enabled and batch versions for both jupyter and database build scripts.<br>Updated and extended documentation. |
| *update-numbaproc* | 20210906 | Implemented a parallel enable/disable keyword for numba enhanced functions.<br>Functions that don't benefit from parallel splitting are hard-coded to parallel=False. |
| *update-tidbits* | 20210908 | Small updates for optional plotting; save/load datacubes, etc.<br>Implemented slurm tidbits; standard scripts can now be run inside interactive sessions on RC systems.<br>Redid how calculations are computed in the CLEDB_BLOS module.<br>Updated constants and ctrlparams classes.|
| *update-CLEDB_BUILD* | 20210915 | Updated database building scripts.<br>Fully implemented the batch run version with $scratch partition use.<br>Updated documentation accordingly.|
| *Update-databugfixes* |20220815| Bug fixes update after running comp inversions. <br> Fixed bugs in obs_qurotate (prepinv module) and cledb_quderotate (proc module). <br> Fixed the cledb_obsderotate (proc module) function. <br> Added angle correction (azimuth derotate) to bphi calculation in cledb_phys (proc module). <br> cledb_getsubset routine (proc module) is optimized and re-written. <br> Introduced the "integrated" control parameter.|   