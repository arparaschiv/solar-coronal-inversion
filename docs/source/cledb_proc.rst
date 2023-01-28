.. _cledb_proc-label:

CLEDB_PROC - Data Analysis \& Inversion
=======================================

**Purpose:**

Three main modules, SPECTRO_PROC, BLOS_PROC, and CLEDB_INVPROC are grouped under the data analysis and inversion module. Based on the 1-line or 2-line input data, two or three modules are called. Line of sight or full vector magnetic field outputs along with plasma, geometric and spectroscopic outputs are inverted here. The algorithm flow and a processing overview is described in the below diagram. 


A. SPECTRO_PROC Function
------------------------

\textbf{Purpose:} Ingests the full spectroscopic prepped data (from OBS\_PREPROCESS) and produces spectroscopic outputs for each input line. Part of the outputs are used downstream in BLOS\_PROC or CLEDB\_INVPROC. This module is heavily dependent on the CLEDB\_PREPINV processing.  Optional sub-modules are envisioned to be integrated into this processing based on upstream instrument processing and retrieved data quality.

~

\textbf{SPECTRO\_PROC main functions:}
\begin{description}
    [font=\normalfont,leftmargin=2.1in,style=multiline]
    \item[CDF\_STATISTICS]
        Performs analysis on the stokes IQUV spectra for each line and computes relevant spectroscopic outputs (see variable description below) by using a non-parametric approach, the analysis of cumulative normal distribution functions.
    \item[(Opt.) ML\_LOSDISENTANGLE]
        To be implemented at a later time. Uses Machine learning techniques for population distributions to disentangle multiple emitting structures along the LOS.
    \item[(Opt.) LEV2CALIB\_WAVE]
        To be implemented at a later time. Higher order wavelength calibration using the spectroscopic profiles. See \citet{Ali+2021} for additional details. This function can couple if the upstream wavelength accuracy of the input observation is missing or is less than 0.01nm.
    \item[(Opt.) LEV2CALIB\_ABSINT]
        To be implemented at a later time, if feasible. Absolute intensity calibration function that produces an additional output, the calibrated intensity in physical units. The approach is not easily automated as it requires a more convoluted and specific planning of the observations to gather the necessary input data.                        
\end{description}

~

\textbf{SPECTRO\_PROC main variables:}
\begin{description}
    [font=\normalfont,leftmargin=1.3in,style=multiline]

    \item[(opt.) sobs\_cal]
        [nx,ny,sn,4] array; optional calibrated level-2 data in intensity and or wavelength units. This is then used by the CDF\_STATISTICS function instead of sobs\_in.                 	
        \item[specout]
		[nx,ny,nline,12] (output)float array; returns the 12 available spectroscopic output calculations, for each input line, and for every pixel location.

		 
    \item[]
    		specout[:, :, :, 0]; Wavelength position of the line core; nm units.
    \item[]
    		specout[:, :, :, 1]; Doppler shift with respect to the theoretical line core; nm units.
    \item[]
        specout[:, :, :, 2]; Doppler shift with respect to the theoretical line core; km s$^{-1}$ units.
    \item[]
    		specout[:, :, :, 3:7]; Intensity at line center wavelength for Stokes I, Q, and U. Stokes V intensity is given as the maximum or minimum counts in the core of the first (left) lobe. Thus, the Stokes V intensity measurement will not match the wavelength position of the Stokes IQU intensities; ADU units or calibrated physical units if LEV2CALIB\_ABSINT is utilized.)
    \item[]
    		specout[:, :, :, 7]; Averaged background intensity outside the line profile for the Stokes I component. Since background counts are independent of the Stokes measurement, we utilize just this one realization; ADU units or calibrated physical units if LEV2CALIB\_ABSINT is used.)
    \item[]
    		specout[:, :, :, 8]; Total line full width at half maximum (FWHM); nm units.
    \item[]
    		specout[:, :, :, 9]; Non-thermal component of the line width. A measure or estimation of the instrumental line broadening/width will significantly increase the accuracy of this determination; nm units.

    \item[pl]
    		specout[:, :, :, 10]; Fraction of linear polarization with respect to the total intensity; dimensionless.                              
    \item[pv]
    		specout[:, :, :, 11];  Fraction of total polarization (linear+circular) with respect to the total intensity; dimensionless.
	\item[Note:]
		Regardless if solving for 1-line or 2-line observations, specout will return two nline dimensions. In the case of 1-line observations, the dimension corresponding to the second line remains just 0 all throughout. The unused dimension can be removed from the upstream script, if needed. This behavior is known and enforced to keep output casting static, speeding up execution when using numba.
\end{description}


B. BLOS_PROC Function
---------------------

\textbf{Purpose:} Implements analytical approximations of \citet{1999ApJ...522..524C} and \citet{2020ApJ...889..109D} to calculate the LOS projected magnetic field strength and magnetic azimuth angle. The module returns two degenerate constrained magnetograph solutions, where the one that matches the sign of the atomic alignment is very precise, along with the less precise "classic" magnetograph formulation.

 This branch is activated for 1-line observations (4 stokes profiles) where the problem is too underdetermined to apply the full inversion database approach using the CLEDB\_INVPROC module.

\textbf{BLOS\_PROC main variables:}
\begin{description}
    [font=\normalfont,leftmargin=1.1in,style=multiline]
    \item[blosout]
        [nx,ny,4] (output) array; line of sight projected magnetic field estimations and magnetic azimuth angle; G units.
    \item[]        
        blosout[:, :, 0] First degenerate constrained magnetograph solution. 
    \item[]         
        blosout[:, :, 1] Second degenerate constrained magnetograph solution.
    \item[]         
        blosout[:, :, 2] "Classic" magnetograph solution. Precision is the average of the two above degenerate solutions. 
    \item[]
        blosout[:, :, 3] Magnetic field azimuth angle derived from the Q and U linear polarization components; Because of using arctan functions to derive, there is a $/pi/2$ degeneracy manifesting in a map.\\ -$\pi$ to $\pi$ range.
    \item[Note:]
    The "classic" magnetograph estimation is less precise than the optimal degenerate constrained magnetograph solution, but more precise than the other. Using just the 4 Stokes IQUV of one observation does not allow us to disentangle which of the two constrained magnetograph solutions is optimal.
                    
\end{description}

.. image:: figs/4_CLEDB_PROC.png
   :width: 800


C. CLEDB_INVPROC Function
-------------------------

\textbf{Purpose:} Main inversion module. CLEDB\_INVPROC compares the preprocessed observations with the selected databases by performing a $\chi^2$ goodness of fit measurement between each independent voxel and the complete set of calculations in the matched database. If CLEDB\_GETSUBSET is utilized, a presorting of the database entries to those that match the direction of lienar polarization is performed. This might presorting might slightly change the final ordering of solutions in certain cases, where some apparently compatible solutions are removed. After the main sorting is performed, the best database solutions are then queried with respect to the physical parameters that gave the matched profiles. 

~

\textbf{CLEDB\_INVPROC main functions:}
\begin{description}
    [font=\normalfont,leftmargin=1.8in,style=multiline]
	\item[CLEDB\_MATCH]
		Matches a set of two full Stokes IQUV observations with a model observation  of the same Stokeq quantities. Matching is done individually for each requested pixel in the input array. This is the most time-consuming function. runtime estimations are provided below.  
    \item[$\Dsh$ CLEDB\_GETSUBSET]
        When enabled, the information encoded in the azimuth is used to reduce the matched database by approximately 1 order of magnitude in terms of calculations. If this subset calculation is disabled from the ctrlparams, execution time in the case of large databases is significantly increased. Although small differences between full and subset sorted database solutions exist, our synthetic data tests did not reveal a situation where performing a calculation with the full database returned a better matching result.  
    \item[$\Dsh$ CLEDB\_PARTSORT]
		A manual function that performs only a partial sort of the array because only a small subset of solutions are usually returned. This increases execution times by a few factors when requesting just few solutions (<50 on 10$^8$ entries databases). The partial sort function is used by both CLEDB\_MATCH and CLEDB\_GETSUBSET functions. In CLEDB\_MATCH, CLEDB\_PARTSORT performs a manual sorting of database entries based on the $\chi^2$ metric. Utilized in CLEDB\_GETSUBSET, CLEDB\_PARTSORT selects for each $\phi$ angle orientation only the most compatible $\Theta$ directions based on the azimuth given by the linear polarization measurements.
    \item[$\Dsh$ CLEDB\_PHYS]
        Returns 9 physical and geometrical parameters corresponding to each selected database index inside the nsearch and maxchisq constraints. these are described below.
    \item[$\Dsh$ CLEDB\_QUDEROTATE]
        Derotates the Q and U components from each selected database entry, in order to make the set of measurements comparable with the original integrated input observation                        
\end{description}

~

\textbf{CLEDB\_INVPROC main variables:}
\begin{description}
    [font=\normalfont,leftmargin=1.3in,style=multiline]
    \item[database]
        list of [ned,nx,nbphi,nbtheta,nline*4] float arrays at input; Individual voxel indexes of the list variables are fed to the CLEDB\_MATCH module. From the database list, only the best matching height entry via db\_enc is passed to CLEDB\_MATCH via database\_sel internal variable. 
	\item[database\_sel]  
        Subset index of the database list that is fed to CLEDB\_MATCH for matching the observation in one voxel. This eases memory shuffling and array slicing operations. This array is reshaped into a 2D  [ned*nx*nbphi*nbtheta(index),nline*4] form. In the case where reduction is selected, the variable is additionally reduced with respect to the number of potential indexes to match. 
    \item[chisq]
        [ned*nx*nbphi*nbtheta,nline*4] float array; Computes the squared difference between the voxel [nline*4] IQUV measurements and each index element of the database [index,nline*4].
    \item[sfound]
        [nx,ny,nsearch,nline*4] (output) float array; returns the de-rotated and matched nsearch IQUV*nline solutions of the database.
        
\newpage           
    \item[invout]
    		[nx,ny,nsearch,11] (output) float array; Main two-line inversion output product. invout contains the matched database index, the $\chi^2$ fitting residuals, and 9 inverted physical parameters, for all nsearch closest matching solutions with respect to the input observation. The 11 parameters follow with individual descriptions.
    	\item[]  
    		invout[nx,ny,nsearch,0] - index; The index of the database entry that was matched at the nsearch rank. The index is used to retrieve the physics that match the observations. 
    	\item[]  
    		invout[nx,ny,nsearch,1] - chisq; The $\chi^2$ residual of the matched database entry.
    	\item[]  
    	    invout[nx,ny,nsearch,2] - edens; Plasma density computed via the ratio of the two ions inverted. This output is applicable for the Fe XIII 1074.68/1079.79 line ratio (same ion). Other line combinations will produce less accurate results due to the relative abundance ratios, that are varying dynamically. For a real-life observation, we do not consider trustworthy the implicit static relative abundance ratios of different ions, resulted from the CHIANTI base tabular data ingested from the ATOM files to build databases. Units are logarithm of number electron density in cm$^{-3}$.
    	\item[]   
    		invout[nx,ny,nsearch,3] - gy; observation apparent height; analogous to yobs variable; $R_\odot$ units.
    	\item[]
    		invout[nx,ny,nsearch,4] - gx; Position of the dominant emitting plasma along the LOS; $R_\odot$ units. 
    	\item[]
    		invout[nx,ny,nsearch,5] - bfield; Magnetic field strength recovered via the ratio of observed stokes V to database Stokes V (computed for B = 1 G); Uses bcalc control parameter; G units.       	
    	\item[]  
    		invout[nx,ny,nsearch,6] - bphi; Magnetic field azimuth angle; 0 to $2\pi$ range.
    	\item[]
    		invout[nx,ny,nsearch,7] - btheta; Magnetic field LOS angle; 0 to $\pi$ range.      	
    	\item[]
    		invout[nx,ny,nsearch,8] - bx; Cartesian projected magnetic field depth/LOS component; G units.
    	\item[]
    		invout[nx,ny,nsearch,9] - by; Cartesian projected magnetic field horizontal component; G units.  	
    	\item[]
    		invout[nx,ny,nsearch,10] - bz; Cartesian projected magnetic field vertical component; G units.
    	\item[Note:]
    	   Regardless of the number of solutions (if any) that are found, the outinv array will return  "0" value arrays, with only the index set to "-1" to keep output data shapes consistent. 	                       
\end{description}
