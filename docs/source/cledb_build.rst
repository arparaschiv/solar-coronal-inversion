.. _cledb_build-label:

CLEDB_BUILD - Database Generation
=================================

**Purpose:**

The ``CLEDB_BUILD`` module is used to generate a database of synthetic IQUV profiles for the four provisioned ions, with a range of density estimations, range of possible :term:`LOS` positions, and all possible magnetic angle configurations, for one magnetic field strength B = 1.  In normal circumstances this module is only run once per system where the inversion is installed. A module diagram is provided in this section.

.. _db_input-label:

CLEDB_BUILD Configuration
-------------------------

Here we describe the scripts included in the config directory.

``DB.INPUT``
	Main configuration file for the database generation. It contains the :term:`physical parameters` configurations for the databases to be generated.

	.. literalinclude:: ../../CLEDB_BUILD/config/DB.INPUT
		:language: c

.. Danger::
	It is critical to keep the same number of parameter decimals and white spaces between the values when modifying the ``DB.INPUT`` configuration file. The automated job-scripts that run the jobs are dependent on precisely reading each entry.

.. _atom-label:

``ATOM.ion``
    This set of files contain the atomic configuration data to be used for calculations. Full level atoms would have a too high computational requirement to use. To avoid this, we use reduced calculations. For example take the Fe XIII lines. The atom configurations are set up as reduced 4-level and 6-transition calculations including the M1 and E1/E2 transitions from upper levels to M1 upper levels. See Fig. 3 of `Casini & Judge, ApJ, 1999 <https://ui.adsabs.harvard.edu/abs/1999ApJ...522..524C/abstract>`_ This level/transition setup mimics the IQUV fluxes from a full level calculation for each of the the selected infrared coronal lines. 

.. Caution::
	Advanced understanding required. In general, users should not modify the ATOM files.
	
.. _hint_ion-label:

``INPUT.ion(a/b)``
	These are input and configuration files that are read when generating databases. The *wlmin* and *wlmax* parameters control which lines described in the ``ATOM.ion`` files are processed. In the case of Fe XIII, a separate ``INPUT.ion`` configuration (a/b) is needed for each line to produce distinct database entries.

	.. hint::
		In the case of Fe XIII, a custom ``INPUT.ion`` configuration with *wlmin* and *wlmax* constraints that includes both lines can be created. This would lead to the synthesis of a direct 2-line database. The :ref:`sdb_preprocess function <sdb_preproc-label>` in the ``CLEDB_PREPINV`` module is provisioned to process such a database configuration. This is an alternate configuration that can be fairly straightforward to implement for a setup aimed at inverting only for the Fe XIII pair. Please note that this is a legacy feature that should not be treated as a default/expected configuration for generating databases.

``IONEQ``
	Ionization equilibrium data from CHIANTI.
	
``GRID.DAT``
	Defines the range and resolution of a CLE simulation. In the case of database building it has no significant functionality and is only required due to CLE's implicit dependency on it's import.

``db"xxxx"\_"arch"``
	Executable CLE binaries for generating databases. *xxxx* is the used version of the CLE Fortran code. arch can be *linux*, *rclinux* or *darwin*. The three different versions are provided in the distribution for cross-platform compatibility.	

	* linux -- Debian compiled

	* rclinux -- CentOs compiled on research computing system.

	* darwin -- mac osx x86 compiled.

	.. Attention::
		Ideally, the *xxxx* version of the CLE code should match its `latest stable release. <https://github.com/arparaschiv/coronal-line-emission>`_  

DB.INPUT Parameters
^^^^^^^^^^^^^^^^^^^

``ny, ymin, ymax``
	Number of y (horizontal) heights in R\ :math:`_\odot` units for which to compute database entries. The ``ny`` heights are spanned between ``ymin`` and ``ymax`` values. Regardless of user input, polarization signal can not computed at this time for R\ :math:`_\odot` < 1 due to the assumptions and interpretation focused on off-limb coronal emission. 

.. attention::
	Observations show that the amount of polarization in Fe XIII drastically decreases with height. One should not normally expect to reasonably recover full Stokes polarization signal at y > 1.5\ :math:`_\odot`.  
	
``ned, elnmin, elnmax``
	Number and range of ambient electron density values for which to compute calculations. ``elnmin`` and ``elnmax`` define a logarithmic range in which to spread the ``ned`` densities. The center of this range is an analytical approximation of a standard electron density expected for a y height above the limb following the Baumbach formulation. See equation 12 and discussion in `Paraschiv & Judge, SolPhys, 2022 <https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract>`_ . For example, at y = 1.1R\ :math:`_\odot` we expect a logarithm of density log(n\ :math:`_e`) ~ 8 cm\ :math:`^{-3}`. Setting ``ned`` = 10, ``elnmin`` = -2 and ``elnmax`` = 2 will generate databases for 10 density values logarithmically scaled between log(n\ :math:`_e`) :math:`\approx` 6 - 10 cm\ :math:`^{-3}`.

.. attention::
	Please keep in mind some potential inversion breaking assumptions. A reasonable density range of log(n\ :math:`_e`) 7-10 is compatible with: 
		
		i. low enough densities so that collisional depolarization becomes unimportant inside the Hanle saturated regime; 
		ii. compatible with expected plasma densities in a standard 1.0-1.5R\ :math:`_\odot` observation range (also remember above point about polarization vs. height).
	
``nx, xmin, xmax``
	Number of x (depth along the :term:`LOS`) positions to compute databases for in R\ :math:`_\odot` units. The ``nx`` positions are linearly spanned between ``xmin`` and ``xmax`` values. 

.. attention::
	Due to geometric considerations, setting ``xmin`` and ``xmax`` values to more than :math:`\pm` 1.0 R\ :math:`_\odot` will most probably not result in practical benefits. This is because a higher 1.5 R\ :math:`_\odot` apparent height, a 1.0 R\ :math:`_\odot` depth would correspond to an actual height above the limb of 1.8 R\ :math:`_\odot`. This is in the more extreme range of the polarization formation vs height issue described above.

``nbphi, bpmin, bpmax``
	Number and range of CLE :math:`\varphi` magnetic :term:`LOS` angles to compute. The ``nbphi`` angles are spread along a ``bpmin`` - ``bpmax``  range set to 0 - 2\ :math:`\pi` by default.
	
``nbtheta, btmin, btmax``	
	Number and range of magnetic CLE :math:`\vartheta` :term:`Azimuth` angles to compute. The ``nbtheta`` range is set to ``btmin`` - ``btmax`` .  By default this is set to a 0 - 1\ :math:`\pi` reduced range due to spherical transformation definitions.

.. Danger::
	Due to how the problem is posed, please do not interchange the maximum ranges between the two magnetic angles, as it would lead to execution errors.	

  

The CLEDB_BUILD Job Script
--------------------------

The *rundb_1line.sh* job script will ingest the ATOM, INPUT, DB.INPUT, etc. files and split the job into available CPU threads. The user is asked for keyboard input on how many threads to use and for which line/ion to generate a database.

.. image:: figs/2_CLEDB_BUILD.png
   :width: 800

The script runs in a Bash shell terminal session. It can handle both Linux and Darwin (OSX) environments. For OSX, an additional dependency is required. Users need to install the GNU implementation of the sed command. The simplest way is to achieve this is by using the homebrew environment:

.. code-block:: bash

	brew install gnu-sed

The job script will split the serial ``ny`` tasks on the requested CPU threads and run in dedicated folders that will be sanitized upon completion, preserving only the output database files and metadata headers. 

Logs for each script ("X") are written in real time and can be checked interactively while the job is running.

.. code-block:: bash

	tail BASHJOB_"X".LOG

A Slurm enabled version, *rundb_1line_slurm*  which has hard-coded choices to be compatible with headless runs is also provided. The parameters need to be checked manually before running. Detailed information about the Slurm enabled routines can be found in the detailed :ref:`readme-slurm-label` section. 

.. note::
	A standalone README-SLURM.MD readme is included in the inversion root directory.

Extensive notes about the parallel job script implementations are found in the detailed :ref:`readme-rundb-label` section.

.. note::
	A standalone README-RUNMD.md readme is included with the ``CLEDB_BUILD`` module.

.. _cledb_output-label:

CLEDB_BUILD Output
------------------

Databases for one up to four of the currently available ions/lines can be constructed by running the job script successively. 

.. tip::
	As long as enough free CPU threads are available, multiple *rundb_1line.sh* jobs can be started simultaneously for **different** ions as there is no storage or computational overlap. 

The output database is written to the storage disk. Each individual line will be written in its dedicated folder. 

.. note::
	Prior to git commit *update-iqud* ``CLEDB_BUILD`` wrote compressed data using a simple float64 :math:`\rightarrow` int16 conversion using a division constant, set to -2.302585092994046e15. Same constant needs to be used when writing but also when reading databases into memory as part of the ``CLEDB_PREPINV`` module. **This approach proved to create numerical instabilities and is currently disabled.**

.. _naming_conv-label:

A database folder hierarchical system is needed in order to ingest the selected database calculations by the ``CLEDB_PREPINV`` module. The folder system is defined as: *element-ionstage_line*.

1. **fe-xiii_1074** 
2. **fe-xiii_1079**
3. **si-x_1430**
4. **si-ix_3934** 

.. note::
	A fifth option for directly writing two line databases for Fe XIII is still preserved as a legacy option as :ref:`described above <hint_ion-label>`. The *.hdr* and *.DAT* database files need to be placed in the main *ctrlparams* :ref:`dbdir key <ctrl_dbdir-label>` without a specific line subfolder.

This convention is used by all three modules of CLEDB.

.. warning:: 
	Running successive jobs for the **same** ion/line will **erase** its database calculations if they exist! 

Individual data stores for each computed height are created to ease I/O operations when reading databases into memory for inverting. A db"xxxx".dat file is generated at each y height in the ``ny`` set, where "xxxx" represents the distance *above the limb* in units of R\ :math:`_\odot` (DB0000.dat corresponds to the solar limb or a height of 1.00R\ :math:`_\odot`). A metadata *db.hdr* file is produced in the individual line directory that contains the range dimensions and parameters applicable to any one database set of files.
 
.. Danger::
	The user should not change the parameter configurations in ``DB.INPUT`` between multiple ion/line runs that should be part of the same database. 

Generating :math:`\sim` 5 :math:`\cdot` 10\ :math:`^8` calculations per line for two lines will occupy :math:`\approx` 32 Gb of disk space with no storage compression.