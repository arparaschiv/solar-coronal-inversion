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

A dedicated readme covering this topic can be :ref:`consulted here <readme-slurm-label>` or as standalone in the mian CLEDB directory. The instructions are provided following the templates set by the `Colorado University Research Computing User Guide <https://curc.readthedocs.io/en/latest/index.html>`_.
