Installation and Run Instructions
=================================

Code Distribution Download
--------------------------

The CLEDB coronal field inversion code distribution is publicly hosted on Github: 

`https://github.com/arparaschiv/solar-coronal-inversion <https://github.com/arparaschiv/solar-coronal-inversion>`_

To create a local deployment use the git clone function:

.. code-block:: bash

   git clone https://github.com/arparaschiv/solar-coronal-inversion

``CLEDB_BUILD`` uses CLE precompiled GNU compatible Fortran binary executable files to generate databases. The module is run by utilizing a Bash script that enables parallel runs of serial computations. Binaries for both Darwin and Linux architectures are provided. More details are found in the :ref:`cledb_build-label` module.

.. Note::
	The CLE FORTRAN source code is not included in this package. It is hosted in a separate repository `https://github.com/arparaschiv/coronal-line-emission <https://github.com/arparaschiv/coronal-line-emission>`_.


A CLEDBenv Python Environment
-----------------------------

The ``CLEDB_PREPINV`` and ``CLEDB_PROC`` modules of CLEDB are written in Python. 
The Anaconda environment system is utilized. Anaconda documentation and installation instructions can be `found here <https://docs.continuum.io/anaconda/install/>`_.

We provide a configuration file ``CLEDBenv.yml`` to create a custom Anaconda environment that groups the CLEDB utilized Python modules briefly :ref:`described above<python_modules-label>`. 

.. literalinclude:: ../../CLEDBenv.yml
   :language: YAML

The configuration file is used to configure the required and optional CLEDB Python packages. In a terminal session you can create the environment via:

.. code-block:: bash

   conda env create -f CLEDBenv.yml
 
After installing all packages the environment can be activated via

.. code-block:: bash

   conda activate CLEDBenv

The user can return to the standard Python package base by running 

.. code-block:: bash

	conda deactivate

If dependency problems arise for any reason, CLEDBenv can be deleted and recreated with the default fixed-version packages from ``CLEDBenv.yml``.

.. code-block:: bash

	conda remove --name CLEDBenv --all

.. Danger::
	The ``CLEDBenv`` anaconda environment installs specific version packages. Cross-compatibility is verified by us. This feature ensures additional codebase stability. Updating the individual Python packages inside the CLEDBenv environment is not recommended and might break code functionality. 


Basic Run Example
------------------

1. Databases can be built with:

	.. code-block:: bash

	   ./CLEDB_BUILD/rundb_1line.sh 

	See detailed database build instructions via the dedicated :ref:`readme-rundb-label` found in the CLEDB\_BUILD directory.

2. Two examples of running the full inversion package (assuming databases are already built) are provided as Jupyter notebooks and/or lab sessions.

	.. code-block:: bash

		./jupyter-lab test_1line.ipynb
		./jupyter-lab test_2line.ipynb

.. Attention::
	Script versions for ``test_1line`` and ``test_2line`` are also available. These are tailored to be used in headless runs. 

Headless Slurm Runs Overview
----------------------------

A few optimizations and modifications are provided in order to ensure a straightforward run of CLEDB on headless systems like research computing clusters. The `Slurm environment <https://slurm.schedmd.com/documentation.html>`_ is utilized. 

Namely:

	* Instructions for resource allocation, installing, and running the inversion in both interactive and batch modes of Slurm research computing setups are provided.
	
	* The database building bash script has a dedicated headless version, *rundb_1line_slurm.sh*, where user options are hard-coded.

	* Pure python test scripts (test\_\*.py) are exported/generated from the Jupyter notebooks (test\_\*.ipynb) to be compatible with batch allocations.

A dedicated readme covering this topic can be :ref:`consulted here <readme-slurm-label>` or as standalone in the main CLEDB directory. The instructions are provided following the templates set by the `Colorado University Research Computing User Guide <https://curc.readthedocs.io/en/latest/index.html>`_.

Example Test Data
-----------------

A number of examples are included to help a user get started with inverting magnetic fields. The test jupyter or python scripts will load different datafiles corresponding to one selected test case.
Some cases are not yet fully implemented or available. The available datafiles can be donwloaded from the links below, or by following the :ref:`Readme.md instructions <readme-main-label>`. The Readme.md file also contains a method for downloading the data using only the terminal for headless systems.

.. hint::
	A DKIST Cryo-NIRSP example will be included once such data becomes available to the community.

	* IQUV test example 1.a 
		Full Stokes IQUV data.
		A `CLE <https://github.com/arparaschiv/coronal-line-emission>` computed forward-synthesis of Fe XIII 1074 and 1079 nm lines using a dipole generator program (See CLE dipolv.f).
		Three independent magnetic dipoles are generated at different positions along the :term:`LOS`. These outputs are combined into a single LOS projected observation.

		.. image:: figs/STOKES_out.png
			:width: 800

		`The 1.a data can be downloaded from gdrive <https://drive.google.com/file/d/1beyDfZbm6epMne92bqlKXcgPjYI2oGRR/view?usp=sharing>`_.

	* IQUV test example 1.b
		Full Stokes IQUV data.
		A `CLE <https://github.com/arparaschiv/coronal-line-emission>`_ computed forward-synthesis of Fe XIII 1074 and 1079 nm lines using a current sheet generator program (See CLE sheet.f).	
		Five simple independent magnetic structures are generated along the LOS to test the algorithm's matching for :term:`LOS` positions.	

		.. image:: figs/los23.png
			:width: 800

		
		.. Attention::
			Two structures are confounded with respect to the LOS leading the inversion to give eroneous results for these locations. This is expected. See `Paraschiv & Judge, SolPhys, 2022 <https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract>`_.
		
		This testcase will be included soon in the example list.

	* IQUV test example 1.c
		Full Stokes IQUV data.
		A MURAM simulation of a dipolar structure at the POS. Fe XIII forward-synthesis via `PyCELp <https://github.com/tschad/pycelp>`_.
		This is a large datafile that is used only for internal testing and can't be shared with the community.

	.. Note::
		For all above examples, a user should expect solutions that are degenerate in pairs of **two** with respect to the LOS position. These need to be properly disambiguated for each observation. A human analysis and decision is required. 

	* IQUD test example 1.d 
		Stokes IQU data. No Stokes V.
		This is an example that unpickles a real observation from CoMP logged on March 27 2012. CoMP is not capable of routinely measuring Stokes V.
		Multiple real-life coronal structures are observed. Because Stokes V is not measured, we do not get access to a analytical solution via the :ref:`BLOS_PROC <blos-label>` module.

		.. image:: figs/comp_iqu.png
			:width: 800

		`The 1.d data can be downloaded from gdrive <https://drive.google.com/file/d/1AdAqIvsiXEV6RK5UiGWcu-1bovs0oOGr/view?usp=sharing>`_.

	* Test data 1.e - Doppler oscillation analysis results for data in 1.d
		This is the additional data that needs to be brought in for obtaining a vector magnetic solution for the CoMP observation offered as part of the 1.d example.
		The two utilized dimensions are ``sobs_dopp[:,:,0]`` and ``sobs_dopp[:,:,1]`` representing respectively the magnetic field strength and the wave angle derived from the Doppler oscillation analysis. The two other dimensions represent :term:`POS` projections of the magnetic field, but are not currently utilized.
		The *test_2line* scripts will just create an empty array when a full Stokes IQUV inversion is requested as in the 1.a - 1.c examples.

		`The 1.e data can be downloaded from gdrive <https://drive.google.com/file/d/1-hPiRRYRS6de_0zWz1k2UU1rIKOEbPOu/view?usp=sharing>`_.

	.. Note::
		For the two datafiles corresponding to the IQUD example, a user should expect solutions that are degenerate in pairs of **four** with respect to the LOS position and the magnetic polarity. Currently a more degenerate solution is retrieved when compared with the full Stokes IQUV inversions. Solutions to further disambiguate IQUD results are currently being trialed. Noteworthy is the fact that the two degeneracies (LOS position and magnetic polarity) are independent with respect to how the problem is posed. Thus, a selection of solutions should not be made as x in set [0,1,2,3] but as x in [1,4] or [2,3] solution subsets for an observed pixel. As mentioned above, these solutions need to be properly disambiguated for each observation. A human analysis and decision is required.

.. Hint::
	A mapping of the magnetic field strength can be obtained from any of the IQUV test 1.a - 1.c cases. These alongside a calculation of the linear polarization azimuth can be fed as a ``sobs_dopp`` observation in a IQUD inversion scheme applied to the same test data. (CLEDB will ignore the Stokes V information in this case). A set of **four** degenerate solutions will be obtained. One subset of **two** solutions will be geometrically identical to the full IQUV inversion output.