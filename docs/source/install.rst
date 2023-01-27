Installation Instructions
=========================

Code Distribution Download
--------------------------

The CLEDB coronal field inversion code distribution is publicly hosted on Github: 

`https://github.com/arparaschiv/solar-coronal-inversion <https://github.com/arparaschiv/solar-coronal-inversion>`_

To create a local deployment use the git clone function:

.. code-block:: bash

   git clone https://github.com/arparaschiv/solar-coronal-inversion

``CLEDB_BUILD`` uses CLE precompiled GNU compatible FORTRAN binary executable files to generate databases. The module is run by utilizing a bash script that enables parallel runs of serial computations. Binaries for both Darwin and Linux architectures are provided. More details are found in the :ref:`cledb_build-label` module.

.. Note::
	The CLE FORTRAN source code is not included in this package. It is hosted in a separate repository `https://github.com/arparaschiv/coronal-line-emission <https://github.com/arparaschiv/coronal-line-emission>`_. The source code is meant to be publicly available and can be requested from the authors.

Python Environment Setup
------------------------

The ``PREPINV`` and ``PROC`` modules of CLEDB are written in Python. 
The Anaconda environment is utilized. Anaconda documentation and installation instructions can be `found here <https://docs.continuum.io/anaconda/install/>`_.

We provide a configuration file ``CLEDBenv.yml`` to create a custom Anaconda environment. 

.. literalinclude:: ../../CLEDBenv.yml
   :language: YAML

The configuration file is used to configure the required and optional CLEDB Python packages. In a terminal session create the environment:

.. code-block:: bash

   conda env create -f CLEDBenv.yml
 
After installing all packages the environment can be activated via

.. code-block:: bash

   conda activate CLEDBenv

The user can return to the standard Python package base by running 

.. code-block:: bash

	conda deactivate

If dependency problems arise, CLEDBenv can be deleted and recreated with the default packages from the .yml.

.. code-block:: bash

	conda remove --name CLEDBenv --all

.. Danger::
	The ``CLEDBenv`` anaconda environment installs specific version packages that are tested. Updating the individual Python packages inside the environment is not recommended and might break functionality. 


Basic Run Examples
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

Utilized Python Modules
-----------------------

For numerical computation efficiency, the CLEDB distribution heavily relies on the Numpy and Numba packages. 
	
* Numpy
	Numpy provides fast vectorized operations on its self implemented-ndarray datatypes. All Python based modules are written in a Numpy-centric way. Functional equivalent pure Python coding is avoided when possible due significantly slower runtimes. Numpy version specific (1.23) documentation is `found here. <https://numpy.org/doc/1.23/>`_
	
* Numba
	Numba implements just in time (JIT) compilation decorators and attempts where possible to perform loop-lifting and scale serial tasks on available CPU threads. Numba has two modes of operation, object-mode and non-python mode. Non-python mode will maximize optimization and runtime speed, but is significantly limited in terms of Python and/or Numpy function compatibility. 

	A Numba enabled implementation can utilize only a small subset of Python and Numpy functions. Significant data sanitation and statically defined function input/output are required in order to enable runtime optimization and parallelization. Due to these sacrifices, coding implementations are not always clear and straightforward. Extensive documentation, Python and Numpy function lists, and examples can be found in the Numba documentation. The version specific (0.56.4) documentation is `available here. <https://numba.readthedocs.io/en/0.53.1/>`_

.. Note::

	The ``CLEDB_PREPINV`` module can only be compiled in object-mode due to disk I/O operations that are not implemented in non-python mode.

* pyyaml
	YAML format library utilized in the ctrlparams class to enable or disable Numba global options. 

* Scipy 
	Used for module fitting and statistics.

* Jupyter, Jupyterlab, Matplotlib and Ipympl
	Optional libraries for data visualization, plotting, widgets, etc.

* Glob, and OS 
	Additional modules used primarily by ``CLEDB_PREPINV`` for I/O operations.

* Time and Sys 
	used during debug runs with high level of verbosity.

* Sphinx and Mist-parser 
	libraries for building documentation and processing markdown files. Disabled as these are not required by the inversion.



