# **CLEDB Coronal Magnetic Field Database Inversion**
[![github](https://img.shields.io/badge/GitHub-arparaschiv%2Fsolar--coronal--inversion-blue.svg?style=flat)](https://github.com/arparaschiv/solar-coronal-inversion)
[![Documentation Status](https://readthedocs.org/projects/cledb/badge/?version=latest)](https://cledb.readthedocs.io/en/latest/?badge=latest)
[![ADS](https://img.shields.io/badge/NASA%20ADS-SoPhys%2C%20V297%2C%20%2063-red)](https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract)



### Repository for **CLEDB** - the **C**oronal **L**ine **E**mission **D**ata**B**ase inversion distribution.

**Authors:** Alin Paraschiv & Philip Judge.  High Altitude Observatory & National Solar Observatory 

**Contact:** arparaschiv "at" ucar.edu; paraschiv.alinrazvan+cledb "at" gmail.com

#### **Main aim:** 
Invert coronal vector magnetic field products from observations of polarized light. 
The algorithm takes arrays of one or two sets of spectro-polarimetric Stokes IQUV observations to derive line of sight and/or full vector magnetic field products.

#### **Applications:** 
Inverting magnetic field information from spectro-polarimetric solar coronal observations from instruments like DKIST Cryo-NIRSP; DL-NIRSP; MLSO COMP/UCOMP.

### **Documentation**

1. Extensive documentation, **including installation instruction, dependencies, algorithm schematics and much more** is available on [CLEDB.READTHEDOCS.IO](https://cledb.readthedocs.io/en/latest/) A git distribution [PDF build](./docs/cledb-readthedocs-io-en-update-iqud.pdf) is also provided.
2. In-depth documentation for the Bash & Fortran parallel database generation module is provided in [README-RUNDB.md](./CLEDB_BUILD/README-RUNDB.md).
3. Installation and usage on RC systems is described in [README-SLURM.md](./README-SLURM.md).
4. This is a beta-level release. Not all functionality is implemented. [TODO.md](./TODO.md) documents updates, current issues, and functions to be implemented in the near future.

### **System platform compatibility**

1. Debian+derivatives Linux x64           -- all inversion modules are fully working.
2. RC system CentOS linux x64             -- all inversion modules are fully working. Additional binary executable is provided. May require local compiling.
3. OSX (Darwin x64) Catalina and Big Sur  -- all inversion modules are fully working; One additional homebrew package required. See [README-CODEDOC.pdf](./codedoc-latex/README-CODEDOC.pdf).
4. Windows platform                       -- not tested.

### **Examples**
Install the CLEDB distribution, generate databases, and update the database save location in the *[ctrlparams.py](./ctrlparams.py)* class, as described in the [README-CODEDOC](./codedoc-latex/README-CODEDOC.pdf).
Afterwards, both 1-line and 2-line implementations of CLEDB can be tested with synthetic data using the two provided Jupyter notebook examples

1. [test_1line.ipynb](./test_1line.ipynb)
2. [test_2line.ipynb](./test_2line.ipynb)


The synthetic CLE test data is [hosted separately here](https://drive.google.com/file/d/1XpBxEwUUyaqYy1NjbVKyCHJhMUKzoV_m/view?usp=sharing).

For terminal only compute systems the test data can be downloaded via the shell interface with the following method:

i. Load the following gdrive wrapper script into your bash window directly, or introduce it in your .bash_alias setup.

    function gdrive_download () {   CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p');   wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2;   rm -rf /tmp/cookies.txt; }

ii. Download the file using its gdrive FILE_ID from the download link (*test data FILE_ID = 1XpBxEwUUyaqYy1NjbVKyCHJhMUKzoV_m*):

    gdrive_download FILE_ID local_path/local_name   (sometimes needs to be run two times to set cookies correctly!)

Note: The script versions of both tests *[test_1line.py](./test_1line.py)* and *[test_2line.py](./test_2line.py)* together with the *[test_cledb_slurm.sh](./test_cledb_slurm.sh)* are slurm enabled to be used for headless RC system runs. 
These offer the same functionality as the notebooks (from which they are directly generated from). See the dedicated [README-SLURM](./README-SLURM.md) for additional information.

Both test examples are expected to fully execute with parallel job spawning via [Numba/JIT](https://numba.readthedocs.io/en/stable/) in a correct installation.


### **Works that fundament and support the CLEDB inversion**

1. [Paraschiv & Judge, SolPhys, 2022](https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract) covered the scientific justification of the algorithm, and the setup of the CLEDB inversion.
2. [Judge, Casini, & Paraschiv, ApJ, 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...912...18J/abstract) discussed the importance of scattering geometry when solving for coronal magnetic fields.
3. [Ali, Paraschiv, Reardon, & Judge, ApJ, 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...932...22A/abstract) performed a spectroscopic exploration of the infrared regions of emission lines available for inversion with CLEDB.   
4. [Dima & Schad, ApJ, 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...889..109D/abstract) discussed potential degeneracies in using certain line combinations. The one-line CLEDB inversion utilizes the methods and results described in this work.
5. [Schiffmann, Brage, Judge, Paraschiv & Wang, ApJ, 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...923..186S/abstract) performed large-scale Lande g factor calculations for ions of interest and discusses degeneracies in context of their results.
6. [Casini & Judge, ApJ, 1999](https://ui.adsabs.harvard.edu/abs/1999ApJ...522..524C/abstract) and [Judge & Casini, ASP proc., 2001](https://ui.adsabs.harvard.edu/abs/2001ASPC..236..503J/abstract) described the theoretical line formation process implemented in CLE, the coronal forward-synthesis code that is currently utilized by CLEDB. 