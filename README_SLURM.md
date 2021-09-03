# **CLEDB - CORONAL FIELD DATABASE INVERSION**
### **SLURM ENABLED RESEARCH COMPUTING INTERACTIVE OR HEADLESS RUNS**

### [solar-coronal-inversion repository on github](https://github.com/arparaschiv/solar-coronal-inversion/)

Detailed instructions for the headless running of the CLEDB inversion.

#### 1. Slurm enabled test scripts 

- test_cledb_slurm.sh

- test_1line.py

- test_2line.py

Note: the *test_1line.py* and *test_1line.py* scripts are plain script versions from the notebook enabled tests. These are directly exported from the Jupyter .ipynb notebooks.
All changes reflected in the notebook examples should be reflected here.

### 2. Installation and run instructions for RC systems

Instructions are pertaining the [CURC system](https://curc.readthedocs.io/en/latest/index.html) and scripts are made compatible with the blanca-nso compute nodes.

Activate the blanca/slurm module with: 

    module load slurm/blanca 

#### 2.a Interactive runs

Start an interactive job:

    sinteractive --partition=blanca-nso --time=01:00:00 --ntasks=2 --nodes=1 --m=48gb

Install the **CLEBD** distribution via git clone in the /projects/$USER/ directory following the instructions in [README-codedoc.PDF](./codedoc-latex/README-CODEDOC.pdf).

update (or create) your .condarc file so that the anaconda environment packages install to your /projects/$USER/ directory instead of /home/$USER/ directory due to lack of storage space.

    pkgs_dirs:
    - /projects/$USER/.conda_pkgs
    envs_dirs:
    - /projects/$USER/software/anaconda/envs

Anaconda install/enable (*this needs to be run for each login*):

    source /curc/sw/anaconda3/latest

Install the **CLEDBenv** environment from the .yml file as outlined (following the instructions in README-codedoc.PDF)
(Install inside the sinteractive run or a compile node following the CURC guidelines. Don't perform the installation from the login node.)

Activate your new environment

    conda activate CLEDBenv

Generate a database:

    module load gcc/10.2.0
    ./CLEDB_BUILD/rundb_1line.sh 

Note: An executable cross compiled on the CURC system is provided and will be used by the script. If libraries are missing, and runs are not executing, please contact us for the CLE source code distribution. The most current CLE distribution is **not yet** publicly hosted.

Update the database save location in the *ctrlparams.py* class, and then run the .py test scripts. 



Everything should be there and work with the exception of remotely connecting to the Jupyter notebooks which refuse to connect.

#### 2.b Batch/headless runs

i. The database generating scripts in CLEDB_BUILD directory have a dedicated headless run script *rundb_1line_slurm.sh* which has slurm headers and all user inputs disabled
The ion to generate the database and the number of cores (that do not exceed xx from #SBATCH --ntasks=xx) need to be manually edited in the script before running. 

This version of the script will perform disk I/O on $SCRATCH partitions. Databases should be moved back to the /projects/$USER/ directories after executing.


The bash *test_cledb_slurm.sh* wrapper script should be used here. It calls one of the two above mentioned .py scripts based on a decision tree.


To be updated when production scripts are ready.