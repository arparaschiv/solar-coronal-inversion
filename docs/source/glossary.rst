Glossary
========

.. glossary::
	Azimuth
		Usually reffering to angles traversing the plane of the sky. These are :math:`\vartheta` or :math:`\Phi_B` depending on geometrical references. 

	ADU
		Arbitrary Data Units; detector calibrated counts when no absolute intensity calibration exists.

	Analytical solutions
	    Frame an inverse problem in a well-understood and reasonably posed mathematical form and approximates a solution. 

	CDF
		Cumulative Distribution Function. Statistical method for interpreting normal distributions. 

	CLE
		Coronal Line Emission FORTRAN spectral synthesis code. It is hosted on `Github <https://github.com/arparaschiv/coronal-line-emission>`_.

	CLEDB
		Coronal Line Emission DataBase Inversion PYTHON algorithm that matches spectropolarimetric observations with CLE generated databases.

	CHIANTI
		atomic database for spectroscopic diagnostics of astrophysical plasmas. See `the documentation <https://www.chiantidatabase.org/>`_.

	:math:`\chi^2` fitting solution
	    Statistical hypothesis to determine whether a variable is likely to come from a specified distribution. The :math:`\chi^2` residual is used to find the closest match to a discrete distribution point.

	Degeneracy
	    When performing an inversion, the degrees of freedom of the problem might not allow to recover an exact mathematical solution. Sets of equivalent solutions inside an inversion metric are called degenerate. e.g., disentangling an angle value knowing that sin a = :math:`\frac{1}{2}`, a is degenerate to either :math:`\frac{\pi}{6}` or :math:`\frac{5\pi}{6}`.

	FWHM
		Full Width at Half Maximum. Measurement of a standard width of a normal distribution.    

	Glob
		This is a `Python library <https://docs.python.org/3/library/glob.html>`_ to process and manipulate os pathnames. 

	Header 
	    Sets of input metadata that accompanies an observation datafile.

	Inversion
	    Mathematical process that starts from the output of a physical 		process and backtraces to recover one or more input variables. In our particular case, we start from output Stokes IQUV profiles and attempt at recovering coronal magnetic fields responsible for producing said profiles.

	JIT
		`Just In Time <https://numba.readthedocs.io/en/stable/reference/jit-compilation.html>`_ compilation decorator from the Numba library package.

	LOS
		Line Of Sight. In CLE references this direction is along the x-axis. The CLE :math:`\varphi` angle traverses this direction. In the observer geometry, the :math:`\Theta_B` angle traverses this direction. 

	Normal distribution
		A Gaussian function, or a bell curve. Probability distribution that is symmetric around a mean value, in which data near the mean are more frequent in occurrence than data far from the mean. 
		
	Numba
		`An open source JIT compiler <https://numba.pydata.org/>`_ that translates a subset of Python and NumPy code into fast machine code. Serial task parallelization and loop-lifting is also available. See `documentation <https://numba.readthedocs.io/en/stable/index.html>`_.

	Numpy
		`Open source library <https://numpy.org/>`_ for fast numeric operations.

	Physical parameters
		A set of observable parameters like density, magnetic field strength, magnetic geometry components, temperature, 3D coordinate position, etc. 

	Physical units
	    Definition of measurement that is calibrated to physically etalonated constants; e.g. intensity in [erg cm\ :math:`^{-2}` s\ :math:`^{-1}` nm\ :math:`^{-1}` sr\ :math:`^{-1}`]

	Pixel
		A 2D representation for a signal integrating area. This is equivalent to a LOS integration of a voxel. This is also the fundamental storage datatype for Python/Numpy arrays. In this document we refer to pixels when discussing data/array elements. 
		
	POS
		Plane Of Sky. In CLE references this direction is correspondent to the zy-plane. The CLE :math:`\vartheta` angle traverses this direction. In the observer geometry, the :math:`\Phi_B` angle traverses this direction.

	Radiative transfer
	    Transfer of electromagnetic radiation through a medium.

	RMS
		Root Mean Square.  The square root of the arithmetic mean of the squares in a set of discrete realizations.

	Slurm
		A computation `worload manager <https://slurm.schedmd.com/documentation.html>`_ used predominantly by research computing clusters.

	Spectroscopic data 
	   Electromagnetic radiation flux spread in individual bins inside an electromagnetic spectral range.

	Spectroscopic emission line
	    Excess flux exceeding background counts at determined spectral positions, occurring when the electrons of an excited atom or molecule move between energy levels.

	Stokes IQUV 
	    A set of values or spectra that describe the polarization state of electromagnetic radiation.

	Stokes I
	    Total intensity of spectroscopic line emission.

	Stokes Q and U
	    Linear polarization components of spectroscopic line emission.

	Stokes V
	    Circular polarization component of spectroscopic line emission.

	Voxel
	    A generalized concept of a pixel. In our case, by voxel we envision 2D projection of a volume inside a square area that contains information about the integral emission along the line of sight. Voxel is used in this document instead of :term:`pixel` when refering to the physical counts recorded inside a spatial integration area of the size of a pixel.                    
