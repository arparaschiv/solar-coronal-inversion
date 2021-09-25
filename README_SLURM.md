# **CLEDB - CORONAL FIELD DATABASE INVERSION**

### [solar-coronal-inversion repository on github](https://github.com/arparaschiv/solar-coronal-inversion/)

### **SLURM ENABLED RESEARCH COMPUTING INTERACTIVE OR HEADLESS RUNS**

Detailed instructions for setting up and running the CLEDB inversion distribution on research computing (RC) systems.

### 1. Slurm enabled test scripts 

- [test_cledb_slurm.sh](./test_cledb_slurm.sh)
- [test_1line.py](./test_1line.py)
- [test_2line.py](./test_2line.py)

Note: the *[test_1line.py](./test_1line.py)* and *[test_2line.py](./test_2line.py)* scripts are plain script versions of the test code.
These are directly exported from the Jupyter .ipynb notebooks.
All changes in the notebook examples should be reflected here.

### 2. Installation and run instructions for RC systems

Instructions are following the [CURC system guidelines](https://curc.readthedocs.io/en/latest/index.html) and scripts are provisioned to be compatible with the blanca-nso compute nodes.

- Activate the blanca/slurm module with: 

        module load slurm/blanca 

#### 2.a Interactive runs

- Start an interactive job:

        sinteractive --partition=blanca-nso --time=01:00:00 --ntasks=2 --nodes=1 --m=12gb

- Install **CLEBD** via git clone in the /projects/$USER/ directory following the instructions in [README-codedoc.PDF](./codedoc-latex/README-CODEDOC.pdf).

- Create or update a .condarc file so that anaconda environments and packages install to your /projects/$USER/ directory instead of /home/$USER/ directory due to lack of storage space.

        pkgs_dirs:
        - /projects/$USER/.conda_pkgs
        envs_dirs:
        - /projects/$USER/software/anaconda/envs

- Anaconda install/enable. This step needs to be run at **each** sinteractive login to enable anaconda.

        source /curc/sw/anaconda3/latest

- Install the **CLEDBenv** anaconda environment from the [CLEDBenv.yml](./CLEDBenv.yml) file. Detailed instructions in [README-codedoc.PDF](./codedoc-latex/README-CODEDOC.pdf).<br>
Note: Install inside the sinteractive run or a compile node following the CURC guidelines. Don't perform the installation from the login node.

- Activate your new environment

        conda activate CLEDBenv

- Generate a database:

        module load gcc/10.2.0
        ./CLEDB_BUILD/rundb_1line.sh 

- Note: A Fortran executable cross compiled on the CURC system with gcc/10.2.0 is provided and will be automatically used by the script. If libraries are missing, and runs are not executing, please contact us for the CLE source code distribution. The most current CLE distribution is **not yet** publicly hosted, but available upon request.

- Update the database save location in the *[ctrlparams.py](./ctrlparams.py)* class, and then run any of the two .py test scripts. 

        python3 test_1line.py
        python3 test_2line.py

- Everything should work (remember to download the test data to the main CLEDB root dir) with the exception of remotely connecting to a Jupyter notebook server spawned inside an sinteractive session (which on CURC refuses to connect). CURC offers dedicated [Jupyter notebook/lab compute nodes](https://curc.readthedocs.io/en/latest/gateways/jupyterhub.html), but beware of how the low resource allocation (usually 1 thread) might interact negatively with the Numba/JIT parallel enabled functions.

#### 2.b Batch/headless runs

- The database generating scripts in CLEDB_BUILD directory have a dedicated headless run script *[rundb_1line_slurm.sh](./CLEBD_BUILD/rundb_1line_slurm.sh)* which has slurm headers and all where user inputs are disabled.
RC resources are asked via the sbatch commands in the script header. The ion to generate the database along with some path variables need to be manually edited in the script before running. This version of the database generation script will perform disk I/O on $SCRATCH partitions, and not on local directories. Databases will be moved back to the /projects/$USER/ directories after computations are finished.

- Call it using sbatch after editing for the ion and paths to generate for each ion (multiple sbatch commends can be run concurrently if resources are available):
        
        sbatch rundb_1line_slurm.sh 

- The bash *[test_cledb_slurm.sh](./test_cledb_slurm.sh)* wrapper script is a starting point for running test/production headless runs via the sbatch command. It provisionally calls one of the two above mentioned .py scripts based on a decision tree.

- The script is to be *updated/finalized* when production runs are ready and data and header ingestion procedures are known.