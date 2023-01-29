.. _inputvars-label:

Input Variables and Parameters
==============================

A. Input Data and Metadata
--------------------------

``header *keys`` 
	Set of input header metadata information that should describe the ``sobs_in`` list variable. Expected keywords with simplified naming are detailed in this section. Detailed keyword information can be found for DKIST observations in the `SPEC_0214 <https://docs.dkist.nso.edu/projects/data-products/en/stable/specs/spec-214.html>`_.

``*keys to crpixn [int]``
    Reference pixel along x y or w(wavelength) direction.

``*keys to crvaln [float]``
    Coordinate value at crpix along x y or w(wavelength) direction.

``*keys to cdeltn [float]``
    float; Spatial (x,y) or spectral (w) platescale sampling along a direction.

``*keys to linpolref [float]``
	(0, 2\ :math:`{\pi})` range; Direction of reference for the linear polarization. This is a physical convention based on the fixed orientation of a spectrograph retarder. `linpolref` = 0 implies the direction is corresponding to a horizontal axis, analogous to the unit circle reference. Direction is trigonometric. The units are in radians. 

``*keys to instwidth [float]``
	Measure of the utilized instrument's intrinsic line broadening coefficient. The units are in nm or km s\ :math:`^{-1}`.

``*keys to nline [int]``
    Number of targeted lines; CLEDVB can accept one-line or two-line observations.  

``*keys to tline [str, nline]``
    String array containing the name of lines to process. Naming convention follows the database directory structure first used by the ``CLEDB_BUILD`` module.

``*keys to xs/naxis1 [int]``
	Pixel dimension of ``sobs_in`` array along the horizontal spatial direction.

``*keys to ys/naxis2 [int]``
	Pixel dimension of ``sobs_in`` array along the vertical spatial direction.

``*keys to ws [int]``
	Pixel dimension of ``sobs_in`` array along the spectral dimension.  

``*keys to skybright [float]``
	Sky brightness measurement used to judge observation quality and rms.

``*keys to grtngba \& grtngang [float]`` 
	The grating order and position; used to find central wavelength of input observation and judge suitability for inverting.   

``keyvals[list]``
	Order is nx, ny, nw, nline, tline, crpix1, crpix2, crpix3, crval1, crval2, crval3, cdelt1, cdelt2, cdelt3, linpolref, instwidth;  This is a variable pack used to more easily feed the necessary parameters to other modules/functions.                        

``sobs_in float array [nline][xs,ys,ws,4]  nline = 1 || 2 for (1-line) or (2-line)``
    ``sobs_in`` is passed as a numba typed list at input. Data are input Stokes IQUV observations of one or two lines respectively. The list will be internally reshaped as a numpy float array of [xs,ys,ws,4] or [xs,ys,ws,8] size.  

.. _ctrl-label:

B. Control Parameters ``ctrlparams.py`` Class
---------------------------------------------

.. literalinclude:: ../../ctrlparams.py
   :language: PYTHON

Python class that unpacks control parameters used in all modules of the inversion setup. This is an editable imported module that users access and modify. The importlib module is used in the example notebooks to reload changes. The yaml import is used to configure NNumba global options.

General Params
^^^^^^^^^^^^^^

``dbdir [string]``
	Directory where the database is stored after being built with CLEDB_BUILD. This is the main directory containing all ions, and not one of the individual ion subdirectories (e.g. fe-xiii_1074, etc.).

``verbose [uint]``
	Verbosity controlling parameter that takes vales 0-3. Levels are incremental (e.g. lev 3 includes outputs from levels 1 and 2) Due to library incompatibilities, enabling higher level verbosity will block Numba optimization of the code. 

	* verbose == 0: Production; silent run.

	* verbose == 1: Debug; prints the current module and operation being run.

	* verbose == 2: Debug; implements warnings for common caveats.
	
	* verbose == 3: Debug; will enable execution timing for selected sections. Numba will fall-back to object-mode.

PREPINV Params
^^^^^^^^^^^^^^
	
``integrated [boolean]``
	To use for calibrated COMP/UCOMP data. In this case, the profiles are integrated across the line sampling points. This parameter defaults to 0 to be applicable to spectroscopic data such as DKIST. 

PROC Params
^^^^^^^^^^^

.. _ctrl_nsearch-label:

``nsearch [uint]``
	Number of solutions to compute and return for each voxel. 
	
``maxchisq [float]``
	Stops searching for solutions in a particular voxel if fitting residuals surpassed this threshold.
	
``gaussfit [uint]``
	Used to switch between CDF fitting and Gaussian parametric fitting with optimization.

	* gaussfit == 0: Process the spectroscopic line parameters using only the CDF method.

	* gaussfit == 1: Fit the line using an optimization based Gaussian procedure. This approach requires a set of 4 guesswork parameters. These are the approximate maximum of the emission (max of curve), the approximate wavelength of the core of the distribution(theoretical center of the line), its standard deviation (theoretical width of 0.16 nm), and an offset (optional, hard-coded as 0).

	* gaussfit == 2: Fit the line using a optimization based Gaussian procedure. In this case, the initial guesswork parameters are fed in from the results of the CDF solution, where the curve theoretically optimizes for a more accurate solution, with sub-voxel resolution.	

.. _ctrl_bcalc-label:
	
``bcalc [uint]``
	Controls how to compute the field strength in the case of 2-line observations.

	* bcalc == 0: Use the field strength ratio of the first coronal line in the list. Only applicable when Stokes V measurements exist; e.g. IQUD is disabled.

	* bcalc == 1: Use the field strength ratio of the second coronal line in the list. Only applicable when Stokes V measurements exist; e.g. IQUD is disabled.
	
	* bcalc == 2: Use the average of field strength ratios of the two coronal lines. Only applicable when Stokes V measurements exist; e.g. IQUD is disabled.
	
	* bcalc == 3: Assigns the field strength from the Doppler oscillation inputs. Only applicable when IQUD is enabled.

.. _ctrl_red-label:

``reduced [boolean]``
	Parameter to reduce the database size before searching for solutions by using the linear polarization measurements. Dimensionality of db is reduced by over 1 order of magnitude, enabling significant sped-ups. Solution ordering might be altered in certain circumstances.

.. _ctrl_iqud-label:

``iqud [boolean]``
	Switches between using Stokes V or Doppler oscillations to compute the magnetic field strength and orientation.

Numba Jit Params
^^^^^^^^^^^^^^^^

``jitparallel [boolean]``
	When Jit is enabled (jitdisable == False), it controls whether parallel loop-lifting allocations are requested, as opposed to just optimize the execution in single-thread-mode. 

``jitcache [boolean]``
	Jit caching for slightly faster repeated execution. Enable only after no changes to \@jit or \@njit functions are required. Otherwise kernel restarts are needed to clear caches. 

``jitdisable [boolean]``
	Debug parameter to control the enabling of Numba just in time compilation (JIT) decorators throughout. Higher level verbosity requires disabling the JIT decorators. This functionality can only be done via Numba GLOBAL flags that need to be written to a configuration file ``.numba_config.yaml``. Any change of this parameter requires a kernel restart.



C. Constants ``constants.py`` Class
-----------------------------------

.. literalinclude:: ../../constants.py
   :language: PYTHON

Python class that unpacks physical constants needed during the inversion. PArameters are mainly utilized by the ``SPECTRO_PROC`` and ``BLOS_PROC`` modules. Ion specific and general atomic and plasma constant parameters are stored herein. The class self-initializes for each required ion providing its *ion specific* parameters in a dynamic fashion.

Physical Consts
^^^^^^^^^^^^^^^^^^

``solar_diam [float*4]``
	Solar diameter in arcsecond, degrees, radians, and steradian units.
	
``l_speed [float]`` 
	Speed of light; SI [m s\ :math:`^{-1}`]  

``kb [float]``
	Boltzmann constant; SI [m\ :math:`^{-2}` kg s\ :math:`^{-2}` K\ :math:`^{-1}`]
	
``e_mass [float]``
	Electron mass; SI [Kg]
	
``e_charge [float]``
	Electron charge; SI [C]
	
``planckconst [float]``
	Planck's constant; SI [m\ :math:`^{-2}` kg s\ :math:`^{-1}`];
	
``bohrmagneton [float]``
	Bohr Magneton; Mostly SI converted to Gauss units [kg m\ :math:`^{-2}` s\ :math:`^{-2}` G\ :math:`^{-1}`]		
		
Ion Specific Consts
^^^^^^^^^^^^^^^^^^^
.. Note::
	4 sets of these constants are provisioned for the four possible ions to invert.

``ion_temp [float]``
	Ion temperature; SI [K]

``ion_mass [float]``
	float; Ion mass; SI [Kg];   

.. _consts_lref-label:

``line_ref [float]``
	Theoretical line core wavelength position; [nm]		

.. Caution::
	Simulation examples might have different set line centers based on the spectral synthesis code used. Doppler shift products might not compute correctly.			

``width_th [float]`` 
	float; Thermal width analytical approximation; [nm]

``F_factor [float]`` 
	Additional factor described by `Dima & Schad, ApJ, 2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...889..109D/abstract>`_. Useful when calculating LOS products in the ``BLOS_PROC`` module;	

``g``\ :sub:`u` \& ``g``\ :sub:`l` ``[float]`` 
	Atomic upper and lower energy levels factors; LS coupling;

``j``\ :sub:`u` \& ``j``\ :sub:`l` ``[float]``	
	Atomic upper and lower level angular momentum terms;

``g``\ :math:`_{eff}` ``[float]``
	LS coupling effective Lande factor;