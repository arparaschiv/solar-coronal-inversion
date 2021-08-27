CLE coronal inversion TODO list:

20210802:

1.  Add the ISSUEMASK setup as outlined in the documentation.

2.  Needed observation Keywords are currently manually implemented, where a synthetic CLE observation is used. 
    Information from observation keywords will need to be ingested after observations will become available.

3.  Develop and use a test case for more convoluted MURAM data.

4.  Implement additional numba compiler flags and options. Make numba active/disabled with a ctrlparam. Implement numba caching.

5.  After more information on input data is obtained, implement the 
    LEV2CALIB_WAVE, LEV2CALIB_ABSINT, and ML_LOSDISENTANGLE functions as outlined in the documentation.

6.  Disentangling multiple LOS contributions in observations.

7.  Refine the noise uncertainty (RMS computations and usage).