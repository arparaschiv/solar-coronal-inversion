# **CLEDB Coronal Magnetic Field Database Inversion**
[![github](https://img.shields.io/badge/GitHub-arparaschiv%2Fsolar--coronal--inversion-blue.svg?style=flat)](https://github.com/arparaschiv/solar-coronal-inversion)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![ADS](https://img.shields.io/badge/NASA%20ADS-SoPhys%2C%20V297%2C%20%2063-red)](https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract)

[![Documentation Status](https://readthedocs.org/projects/cledb/badge/?version=latest)](https://cledb.readthedocs.io/en/latest/?badge=latest)
![Run Tests](https://github.com/arparaschiv/solar-coronal-inversion/actions/workflows/test.yml/badge.svg)


### Repository for **CLEDB** - the **C**oronal **L**ine **E**mission **D**ata**B**ase inversion.

**Authors:** Alin Paraschiv and Philip Judge. National Solar Observatory & High Altitude Observatory

**Contact:** arparaschiv "at" nso.edu

#### **Main aim:**
Invert coronal vector magnetic field products from observations of polarized light.
The algorithm takes arrays of one or two sets of spectro-polarimetric Stokes IQUV observations to derive line of sight and/or full vector magnetic field products.

#### **Applications:**
Inverting magnetic field information from spectro-polarimetric solar coronal observations from instruments like DKIST Cryo-NIRSP; DL-NIRSP; MLSO CoMP/UCoMP.

### **Documentation**

1. Extensive documentation, **including installation instruction, dependencies, algorithm schematics and much more** is available on [CLEDB.READTHEDOCS.IO](https://cledb.readthedocs.io/en/latest/) A git distribution [PDF build](./docs/cledb-readthedocs-io-en-update-iqud.pdf) is also provided.
2. In-depth documentation for the Bash & Fortran parallel database generation module is provided in [README-RUNDB.md](./CLEDB_BUILD/README-RUNDB.md).
3. Installation and usage on RC systems is described in [README-SLURM.md](./README-SLURM.md).
4. This is a beta-level release. Not all functionality is implemented. [TODO.md](./TODO.md) documents updates, current issues, and functions to be implemented in the near future.

### **System platform compatibility**

1. Debian+derivatives Linux x64           -- all inversion modules are fully working.
2. RC system CentOS linux x64             -- all inversion modules are fully working. An additional binary executable is provided. May require local compiling.
3. OSX (Darwin x64) Catalina and Big Sur  -- all inversion modules are fully working; One additional homebrew package required. See [README-RUNDB](./CLEDB_BUILD/README-RUNDB.md).
4. Windows platform                       -- not tested.

### **Examples**
Install the CLEDB distribution, generate databases, and, if needed, update the database save location in the *[ctrlparams.py](./ctrlparams.py)* class, as described in the [CLEDB.READTHEDOCS.IO](https://cledb.readthedocs.io/en/latest/) .

The new PyCELP database generation tool is recommended. It is more precise, but requires some computational resources for calculations. [A default PyCELP generated database can be found here to help get started (33Gb download)](https://drive.google.com/file/d/1Rehk5jThUzE_9LyoQpi6lJcGlomFIOPy/view?usp=sharing). The two database line folders need to be extracted to the CLEDB_BUILD directory.

Afterward, both 1-line and 2-line implementations of CLEDB can be tested with synthetic data using the two provided Jupyter notebook examples

1. [test_1line.ipynb](./test_1line.ipynb)
2. [test_2line.ipynb](./test_2line.ipynb)

The test data are hosted separately. These are called by enabling the corresponding 1.- 4. cells in the test notebooks and scripts. See the [documentation](https://cledb.readthedocs.io/en/latest/install.html) for details regarding the included datafiles.

- [1. synthetic CLE 3 dipole "observation"](https://drive.google.com/file/d/1beyDfZbm6epMne92bqlKXcgPjYI2oGRR/view?usp=drive_link).
- [2. Synthetic PyCELP "observation" from MURAM cube](https://drive.google.com/file/d/1G03zXcyREF5BBZtay1P1sjcBlZyVNDF4/view?usp=drive_link)
- [3. DKIST Cryo-NIRSP line-integrated observation](https://drive.google.com/file/d/1o65wMbcmobTVHOSnEPOQhJmG4hGk3Hyt/view?usp=drive_link)
- [4. CoMP IQUD-only line-integrated observation](https://drive.google.com/file/d/1P1XvRy7VXehkJvmx_zVmhzlSyacOVwtb/view?usp=drive_link).
- [5. Additional CoMP and UCoMP IQUD examples associated to publications](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y6M7HJ)

For terminal-only compute systems, test data can be downloaded directly from Google Drive using gdown.

i Install gdown (requires Python 3.10 or later):

    conda install -c conda-forge gdown

ii. Download the file using its gdrive FILE_ID from the download link (precompiled database FILE_ID = 1Rehk5jThUzE_9LyoQpi6lJcGlomFIOPy):

    gdown 1Rehk5jThUzE_9LyoQpi6lJcGlomFIOPy -O local_path/local_name

Note: The script versions of all tests *[test_1line.py](./test_1line.py)* and *[test_2line.py](./test_2line.py)* together with the *[test_cledb_slurm.sh](./test_cledb_slurm.sh)* are slurm enabled to be used for headless RC system runs.
These offer the same functionality as the notebooks (from which they are directly generated from). See the dedicated [README-SLURM](./README-SLURM.md) for additional information.

Both test examples are expected to fully execute with parallel/m,ultithreaded job spawning in a correct installation.

### **Contributions**

We welcome contribution ideas and even implementations of new functionalities and optimizations to be included in CLEDB. This can be done through a pull-merge request or by contacting the developers directly to discuss your plans and ideas.
The developers will strive to accept and implement contributions as long as they fit within the scope of the software and are adhering to the code of conduct.

### **Acknowledgement: Works that fundament and support the CLEDB methodology**


1. [Paraschiv, Yang, Lacatus, & Molnar, ApJL, 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260935100P/abstract) demonstrates the applicability of the IQUD inversion methodology to both CoMP and UCoMP observations. (in press)
2. [Paraschiv, ApJ, 2026](https://ui.adsabs.harvard.edu/abs/2026ApJ..1008...29P/abstract) provides the theoretical basis of the IQUD approach and performs a comprehensive statistical analysis demonstrating that the IQUD inversion methodology, has an accuracy comparable to that of Stokes IQUV spectropolarimetry.
3. [Paraschiv & Judge, SolPhys, 2022](https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract) covered the scientific justification of the inversion algorithm, and the setup of the CLEDB inversion.
4. [Judge, Casini, & Paraschiv, ApJ, 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...912...18J/abstract) discussed the importance of scattering geometry when solving for coronal magnetic fields.
5. [Ali, Paraschiv, Reardon, & Judge, ApJ, 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...932...22A/abstract) performed a spectroscopic exploration of the infrared regions of emission lines available for inversion with CLEDB.
6. [Dima & Schad, ApJ, 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...889..109D/abstract) discussed potential degeneracies in using certain line combinations. The one-line CLEDB inversion utilizes the methods and results described in this work.
7. [Schiffmann, Brage, Judge, Paraschiv & Wang, ApJ, 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...923..186S/abstract) performed large-scale Lande g factor calculations for ions of interest and discusses degeneracies in context of their results.
8. [Casini & Judge, ApJ, 1999](https://ui.adsabs.harvard.edu/abs/1999ApJ...522..524C/abstract) and [Judge & Casini, ASP proc., 2001](https://ui.adsabs.harvard.edu/abs/2001ASPC..236..503J/abstract) described the theoretical line formation process implemented in CLE, the coronal forward-synthesis code that is currently utilized by CLEDB.
<!--
3. [Yang, Rempel, Gibson, & de Toma, ApJS, 2026](https://ui.adsabs.harvard.edu/abs/2026ApJS..283....3Y/abstract) use forward modeling to demonstrate that two-dimensional (2D) coronal seismology can recover the line-of-sight (LOS) emissivity-weighted magnetic field direction and strength accurately.
4. [Yang, Tian, Tomczyk, Liu, Gibson, Morton, & Downs, Science, 2024](https://ui.adsabs.harvard.edu/abs/2024Sci...386...76Y/abstract) uses coronal heliseismology applied to observations from the Upgraded Coronal Multi-channel Polarimeter, to provide 114 magnetograms of the global corona above the solar limb spanning ~8 months. 
-->
