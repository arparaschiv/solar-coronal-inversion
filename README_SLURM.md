# **CLEDB CORONAL FIELD DATABASE INVERSION - SLURM RESEARCH COMPUTING INTERACTIVE OR HEADLESS RUNS**
### [solar-coronal-inversion repository on github](https://github.com/arparaschiv/solar-coronal-inversion/)

Detailed instructions for the headless running of the CLEDB inversion.

#### 1. Slurm enabled test scripts 

- test_cledb_slurm.sh

- test_1line.py

- test_2line.py

Note: the *test_1line.py* and *test_1line.py* scripts are plain script versions from the notebook enabled tests. These are directly exported from the Jupyter .ipynb notebooks.
All changes reflected in the notebook examples should be reflected here.

### 2. Installation and run instructions RC systems

Instructions are pertaining the [CURC system](https://curc.readthedocs.io/en/latest/index.html) and scripts are made compatible with the blanca-nso compute nodes.

Activate the blanca/slurm module with: 
    module load slurm/blanca 



#### 2.a interactive runs

Start an interactive job:
    sinteractive --partition=blanca-nso --time=01:00:00 --ntasks=2 --nodes=1 --m=48gb

Install the **CLEBD** distribution via git clone in the /projects/$USER/ directory (following the instructions in README-codedoc.PDF)

update (or create) your .condarc file so that the anaconda environment packages install to your /projects/$USER/ directory instead of /home/$USER/ directory due to lack of storage space.
    pkgs_dirs:
    - /projects/$USER/.conda_pkgs
    envs_dirs:
    - /projects/$USER/software/anaconda/envs

Anaconda install/enable (*this needs to be run for each login*):
    source /curc/sw/anaconda3/latest

Install the **CLEDBenv** environment from the .yml file as outlined (following the instructions in README-codedoc.PDF)
(Install inside the sinteractive run as to the CURC guidelines. Don't perform the installation from the login node.)

Generate a database and then run the standard test notebooks. Jupyter and everything should be there and work.

#### 2.b headless runs

The database generating scripts in CLEDB_BUILD directory have a dedicated headless run script *rundb_1line_slurm.sh* which has slurm headers and all manual entries disabled
The ion to generate the database and the number of cores (that do not exceed xx from #SBATCH --ntasks=xx) need to be manually edited in the script before running.

The bash *test_cledb_slurm.sh* script should be used here. It calls one of the two .py scripts based on a decision tree.


To be updated when production scripts are ready