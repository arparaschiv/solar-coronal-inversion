Installation and Run Instructions
=================================

Code Distribution Download
--------------------------

The CLEDB coronal field inversion code distribution is publicly hosted on Github:

`https://github.com/arparaschiv/solar-coronal-inversion <https://github.com/arparaschiv/solar-coronal-inversion>`_

To create a local deployment, use the git clone function:

.. code-block:: bash

   git clone https://github.com/arparaschiv/solar-coronal-inversion


A CLEDBenv Python Environment
-----------------------------

The ``CLEDB_PREPINV`` and ``CLEDB_PROC`` modules of CLEDB are written in Python.
The Anaconda environment system is utilized. Anaconda documentation and installation instructions can be `found here <https://docs.continuum.io/anaconda/install/>`_.

We provide a configuration file ``CLEDBenv.yml`` to create a custom Anaconda environment that groups the CLEDB  modules that are briefly :ref:`described above<python_modules-label>`.The environment file can be used  to configure and update the main and optional Python packages required by CLEDB.

.. literalinclude:: ../../CLEDBenv.yml
   :language: YAML

In a terminal session the environment can be installed:

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
	The ``CLEDBenv`` anaconda environment installs specific version packages. This feature ensures better codebase stability. Updating the individual Python packages inside the CLEDBenv environment is not recommended because it might break code functionality.


Basic Run Example
------------------

Move to the CLEDB working directory

    .. code-block:: bash

        cd ./solar-coronal-inversion

1. Databases can be built with either PyCELP or CLE command below:

	.. code-block:: bash

	   ./CLEDB_BUILD/rundb_1line_with_*.sh

	See detailed database build instructions via the dedicated :ref:`readme-rundb-label` found in the CLEDB\_BUILD directory.

``CLEDB_BUILD`` can utilize:

- CLE precompiled GNU compatible Fortran binary executable files to generate databases. The module is run by utilizing a Bash script that enables parallel runs of serial computations. Binaries for both Darwin and Linux architectures are provided. More details are found in the :ref:`cledb_build-label` module.

or

- PyCELP python script to generate databases. This module is executed by running the python script with the same basic configuration as CLE. Parallelization is enabled. No binaries are required, but the PyCELP and CHIANTI databases need to be locally installed. The database build readme provides basic instructions on how to build a pycelp database. For simplicity, a precompiled database can be downloaded using the link in the main readme.


    PyCELP database building requires the instalation of the respective package from git:

.. code-block:: python

  conda activate CLEDBenv
  git clone https://github.com/tschad/pycelp.git
  cd pycelp
  python setup.py develop

Extended instructions can be found in the `PyCELP <https://github.com/tschad/pycelp>`_ repository.


.. Note::
	The CLE FORTRAN and PyCELP source codes are not included in the CLEDB package. These are maintained and hosted in separate repositories: `https://github.com/tschad/pycelp <https://github.com/tschad/pycelp>`_ and `https://github.com/arparaschiv/coronal-line-emission <https://github.com/arparaschiv/coronal-line-emission>`_.


A precompiled PyCELP database can be `found here (33Gb download) <https://drive.google.com/file/d/1Rehk5jThUzE_9LyoQpi6lJcGlomFIOPy/view?usp=sharing>`_.




2. Two examples of running the full inversion package (assuming databases are already built in the 2-line case) are provided as Jupyter notebooks and/or lab sessions.

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

Example Datacubes
-----------------

A number of examples are included to help a user get started with inverting coronal fields with CLEDB. The test jupyter or python scripts will load different datafiles corresponding to one selected test case.
Some cases are not yet fully implemented or available. The available datafiles can be donwloaded from the links below, or by following the :ref:`Readme.md instructions <readme-main-label>`. The Readme.md file also contains a method for downloading the data using only the terminal for headless systems.


* 1. CLE IQUV test example
	Full Stokes synthetic IQUV data to test CLE generated databases.
	A `CLE <https://github.com/arparaschiv/coronal-line-emission>` computed forward-synthesis of Fe XIII 1074 and 1079 nm lines using a dipole generator program (See CLE dipolv.f).
	Three independent magnetic dipoles are generated at different positions along the :term:`LOS`. These outputs are combined into a single :term:`LOS` projected observation.

	.. image:: figs/STOKES_out.png
		:width: 800

	`The data can be downloaded from gdrive <https://drive.google.com/file/d/1beyDfZbm6epMne92bqlKXcgPjYI2oGRR/view?usp=sharing>`_.

	.. Attention::
		The structures are confounded with respect to the LOS leading the inversion to give erroneous results for these locations. This is expected. See `Paraschiv & Judge, SolPhys, 2022 <https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract>`_.

    .. Warning::
        This example is created using CLE and is available for testing CLE databases. The spectral synthesis assumptions vary between CLE and PyCELP. In regions of high density, the inversion will lead to significant changes. One of the dipolar structures will not invert accurately invert when using PyCELP databases. With PyCELP databases, setting the control parameter to 1e6 will force a result output, but care needs to be used when interpreting.


* 2. PyCELP and MURAM IQUV test example
	Full Stokes IQUV data to test PyCELP generated databases.
	A MURAM simulation of a dipolar structure at the :term`POS`. Fe XIII forward-synthesis via `PyCELP <https://github.com/tschad/pycelp>`_. This synthetic observation was analyzed and described by `Schad & Dima, SolPhys, 2021 <https://ui.adsabs.harvard.edu/abs/2021SoPh..296..166S/abstract>`_.
	This is a large datafile.

	.. image:: figs/muram_iquv.png
		:width: 800

    `The data can be downloaded from gdrive <https://drive.google.com/file/d/1G03zXcyREF5BBZtay1P1sjcBlZyVNDF4/view?usp=sharing>`_.


* 3. DKIST Cryo-NIRSP integrated data test example.
    Full Stokes line-integrated IQUV data.
    This example utilizes a real observation from DKIST Cryo-NIRSP from on March 23 2024. A preliminary processed observation of spectro-polarimetric Stokes IQUV measurements.

	.. image:: figs/cryonirsp_iquv.png
		:width: 800

	`The data can be downloaded from gdrive <https://drive.google.com/file/d/1o65wMbcmobTVHOSnEPOQhJmG4hGk3Hyt/view?usp=sharing>`_.

.. Caution::
    The 3.a datafiles are line-integrated IQUV datasets resulting from the analysis tutorial processing provided by the DKIST Science team. `The data analysis tutorial can be found here <https://bitbucket.org/dkist-community-code/cryonirsp-notebooks/src/main/first_release_specFitting/>`_ This processing is just a conceptual example of analysis steps and not a careful science-ready processing. Neither the input data nor the output CLEDB products are scientifically validated and science ready.

.. Important::
    Processing of full spectro-polarimetric DKIST data is not yet included in CLEDB due to the complexities of manually tuning the processing of each dataset. It is recommended to pursue an analysis following the steps described in the public DKIST Cryo-NIRSP tutorial linked above, and then feed the resulting integrated Stokes IQUV profiles after accounting for the photosphere, tellurics, crosstalk, and interference fringes.


* 4. CoMP IQUD test example
	Stokes IQU data. No Stokes V.
	This example utilizes a real observation from CoMP from on March 27 2012. CoMP is not capable of routinely measuring Stokes V. UCoMP ingestion is also implemented in CLEDB.
	Multiple real-life coronal structures are observed. Because Stokes V is not measured, we do not get access to a 1-line analytical solution via the :ref:`BLOS_PROC <blos-label>` module.

	.. image:: figs/comp_20120327_iqu.png
		:width: 800

	Doppler oscillation analysis B_POS is required in addition to the CoMP/UCoMP observations for infering a vector magnetic solution. This data and methods of inferring POS magnetic maps from Doppler oscillations are described in `Yang et. al, Sci. 2020 <https://ui.adsabs.harvard.edu/abs/2020Sci...369..694Y/abstract>`_, and `Yang, et. al, Sci. 2024 <https://ui.adsabs.harvard.edu/abs/2024Sci...386...76Y/abstract>`_.
	We utilize ``sobs_dopp`` that encodes the :term:`POS` projected magnetic field strength and the wave propagation angles. The *test_2line* scripts will create an empty ``sobs_dopp`` array when a full Stokes IQUV inversion is pursued due to the implemented numba optimization schemes.

	.. image:: figs/comp_20120327_blos.png
		:width: 200

	`The IQU + Doppler oscillation B_POS data can be downloaded from gdrive <https://drive.google.com/file/d/1P1XvRy7VXehkJvmx_zVmhzlSyacOVwtb/view?usp=sharing>`_.


*5 Additional CoMP and UCoMP IQUD inversion examples are part of the replication data for the first 3D magnetic field inference publication ('Paraschiv, Yang, Lacatus, & Molnar, ApJL, 2026 <https://ui.adsabs.harvard.edu/abs/2026ApJL/abstract>'_), and are hosted through a Harvard Dataverse repository ('doi:10.7910/DVN/Y6M7HJ <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y6M7HJ>'_).

.. Note::
	For all IQUV examples, a user should expect solutions that are degenerate in pairs of **two** with respect to the LOS position. These need to be additionally disambiguated for each observation. An analysis of the inversion outputs and decision is required.

.. Note::
	In the case of the IQUD example, a user should expect solutions that are degenerate in pairs of **four** with respect to the LOS position and the magnetic polarity. Currently, a more degenerate solution is retrieved when compared with the full Stokes IQUV inversions. Solutions to further disambiguate IQUD results are currently being trialed. Noteworthy is the fact that the two degeneracies (LOS position and magnetic polarity) are independent with respect to how the problem is posed. Thus, a selection of solutions should not be made as x in set [0,1,2,3] but as x in [1,4] or [2,3] solution subsets for an observed pixel. As mentioned above, these solutions need to be properly disambiguated for each observation. A manual analysis and decision is required.

.. Hint::
	A mapping of the magnetic field strength can be obtained from any of the IQUV test 1. - 3. cases. These alongside a calculation of the linear polarization azimuth can be fed as a ``sobs_dopp`` observation in a IQUD inversion scheme applied to the same test data or particular observations. (CLEDB will ignore the Stokes V information in this case). A set of **four** degenerate solutions will be obtained. One subset of **two** solutions will be geometrically identical to the full IQUV inversion output.


