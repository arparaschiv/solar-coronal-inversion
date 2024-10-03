# **CLEDB Parallel Database Generator**
[![github](https://img.shields.io/badge/GitHub-arparaschiv%2Fsolar--coronal--inversion-blue.svg?style=flat)](https://github.com/arparaschiv/solar-coronal-inversion)

README for performing PyCELP or CLE database calculations on multiple CPU threads.

Contact: Alin Paraschiv (arparaschiv at nso edu)

#### **History for the BUILD module:**

- ARP: 20210617 - initial release. 
- ARP: 20210827 - Added a Slurm enabled version of the script for batch jobs on RC systems.
- ARP: 20210915 - Rewrote the thread scaling to allocate tasks uniformly across threads; Both interactive and batch scripts now can utilize RC Slurm capabilities. The interactive version can only use Slurm allocated resources inside interactive jobs. The batch dedicated version can utilize scratch directories; It copies final outputs in a user's project directory after finalizing tasks.
- ARP: 20221222 - Updated both scripts to fix an error with calculating the optimal heights that are scaled across available nodes.
- ARP: 20240730 - Overhaul of the database building functionality. A new implementation of the CLEDB BUILD module using PyCELP is now provided.

#### **SCOPE:**

This is a script implementation that launches separate parallel processes for building Stokes IQUV databases as part of the CLEDB_BUILD module.
Three versions are provisioned:

1. *[rundb_1line_with_PyCELP.py](./rundb_1line_with_PyCELP.py)*        -- PyCELP:  For local **interactive** and **batch** runs.
3. *[rundb_1line_with_CLE.sh](./rundb_1line_with_CLE.sh)*              -- DEPRECATED -- CLE: For local **interactive** runs (Slurm interactive session compatible).
4. *[rundb_1line_with_CLE_batch.sh](./rundb_1line_with_CLE_batch.sh)*  -- DEPRECATED -- CLE: For **batch** and/or headless runs.

#### **INSTALL and USAGE:**
- PyCELP database building requires three components:

    a. The *CLEDBenv* conda python environment to be installed. Instructions: [CLEDB.READTHEDOCS.IO](https://cledb.readthedocs.io/en/latest/install.html#a-cledbenv-python-environment)

    b. PyCELP will need to be installed separately in the *CLEDBenv* environment, following the instructions [on Github](https://github.com/tschad/pycelp/tree/main) (In this casedo not create a separate environment as recommended there). PyCELP can not be automatically included in the *CLEDBenv* environment at this moment.

    c. The latest available version CHIANTI database to be downloaded from the [CHIANTI website](https://www.chiantidatabase.org/chianti_download.html). Default CHIANTI download folder in CLEDB is [./CLEDB_BUILD/config/PyCELP/](./CLEDB_BUILD/config/PyCELP/). Otherwise, the XUVTOP path in the [rundb_1line_with_PyCELP.py](./rundb_1line_with_PyCELP.py) script will need updating to where you installed CHIANTI.

- make sure the scripts you plan using are executable:

        chmod u+x rundb_1line_with_XXXX.yy

- With PyCELP: run any type of job via (see notes below about options):

        conda activate CLEDBenv
        python rundb_1line_with_PyCELP.py in1 in2 in3
        or
        nohup python rundb_1line_with_PyCELP.py in1 in2 in3 & (This frees the terminal and appends all output to a text file called nohup in the directory from which the script is run.)

- With CLE (deprecated): run interactive jobs (after starting the interactive node; see README_SLURM) and batch/headless jobs with either:

        ./rundb_1line_with_CLE.sh

        or

        sbatch rundb_1line_with_CLE_batch.sh

- (Optional for OSX) Install gnu-sed (See notes below). OSX might also have issues with running executables ("cannot execute binary file") Try changing permissions with xattr.

        brew install gnu-sed

        xattr -d com.apple.quarantine /path/to/file

- (Optional) for interactive jobs on RC systems, the correct modules may need to be preloaded in order for scripts to execute. 

        module load slurm/blanca
        module load gcc/10.2.0                (gcc is preloaded automatically in the batch version of the script.)


#### **NOTES:**
- ** NEWLY COMPLETED RUNS WILL DELETE/OVERWRITE PREVIOUSLY COMPUTED CALCULATIONS AND LOGS IN THE CORRESPONDENT SUBFOLDER**

- The scripts are configured to produce one line database outputs. All atomic data for the four ions of interest along with the configuration files are available in the *config* directory. This setup selects the relevant inputs automatically.

- A database configuration file is [CLEDB_BUILD/DB.INPUT](./CLEDB_BUILD/DB.INPUT) that configures the database number of calculations (parameter resolution) that is read by either PyCELP or CLE tools.

Production Databases generated via PyCELp calculation:

- The *[rundb_1line_with_PyCELP.py](./rundb_1line_with_PyCELP.py)* script can be run straight using default values or by specifying two direct parameter inputs, in1, in2, and in3.

    - in1 is a mandatory input, while in2 and in3 are optional.

    - in1 -- The desired line to calculate. Options 1-4 correspond to:
        1:    FE XIII 1074.7nm
        2:    FE XIII 1079.8nm
        3:    Si X    1430.1nm
        4:    Si IX   3934.3nm

    - in2 -- (optional) The number of CPU threads to use. Valid options are 1 to n - 4 threads, where n represents the available system threads. If the number is bigger than available threads, the script will use n-4 threads to leave room for cpu task overhead.  By default, the script will scale n-4 parallel threads to run calculations.

    - in3 -- (optional) The number of atomic levels to include in calculations. The script is internally configured to run with 25 atomic levels. Although this ensures a fast execution for the default DB.INPUT configuration, the computed databases will not be as accurate for precision inversion calculations. About 80 levels minimum are required for a quantitative analysis level database, although this is rather computationally demanding. A precompiled 80 level database can be downloaded from the link provided in the main CLEDB readme.

Alternative databases  generated via CLE calculations --**Deprecated**--
- The *[rundb_1line_with_CLE.sh](./rundb_1line_with_CLE.sh)* script requires two manual keyboard user inputs.

    i. select how many CPU threads to use;

        Hi
        You have xx CPU threads available.
        How many to use?

    ii. which ion/line to compute. Each ion/line will create its own subfolder in the directory structure to store computations.

        Please indicate the line to generate. Options are:
        1:    FE XIII 1074.7nm
        2:    FE XIII 1079.8nm
        3:    Si X    1430.1nm
        4:    Si IX   3934.3nm

- The batch *[rundb_1line_CLE_slurm.sh](./rundb_1line_CLE_slurm.sh)* script has no keyboard inputs, but has manually defined variables that a user can edit to control the ions to generate.

- Most directory and file pointers are dynamically linked to the CLEDB distribution directory. Local runs should run without interference. Some directory/system variables are defined to be compatible with the CURC system (scratch, project, etc.). These may need to be updated for different systems.

- The CLE databases are more compact in terms of physical size and require less resources to be generated. The parameter space and accuracy of these calculations (due to very low number atom levels that can included) are significantly lower than of the PyCELP calculations. Great caution is required for interpretation of matching solutions. Also, due to a different implementation, using these databases will lead to longer computation times of the inversion.

#### *Additional CLE specific notes - DEPRECATED functionality*

- The ./rundb_1line_with_CLE_XXX.sh scripts implement an external parallelization allocation and will wait for all thread tasks to finish before exiting.
Due to limitation in CPU process ID (PID) tracking, the user is not notified in order of threads finalizing, but in the order they were scheduled. e.g. if thread 2 finishes before thread 0, the user will find out only after thread 0 and thread 1 finish. A bug might manifest if a new unrelated task is scheduled with the same PID as one of the runs, but this should not occur in normal circumstances. If such a case occurs, a tail of the logs will verify that everything went well and scripts can be exited manually.

- The number of Y-heights to calculate between the ymin and ymax ranges are not always a multiple of the number of CPU threads.
The scripts will efficiently scale the tasks on the available threads. If you request less tasks (via [DB.INPUT](./config/DB.INPUT)) than threads (via keyboard or sbatch), the script will not utilize all pre-allocated resources.

- The script heavily relies on the SED function. 
SED has different implementations on Linux (GNU) vs mac (BSD) which makes commands not be directly correspondent. A function wrapper *SEDI* that disentangles GNU vs BSD syntax is provided in the scripts. OSX users need to install a gnu implementation of sed (gnu-sed) for the script to be portable between systems (via the gsed command).

        brew install gnu-sed

- The script cuts and appends midline on the DB.INPUT file, to set the ymin and ymax ranges for each CPU thread.
The number of decimals for all variables and 3 spaces in between them need to be kept in the configuration file in order to not introduce bugs.

- Executables (dbxxx) need to be build (from CLE) on the current architecture: ELF(linux) or Mach-O(OSX) 
If non-correct executables are called a "cannot execute binary file" error is produced. Architecture can be checked with the *file* command. The configuration deduces the OS in use and selects and uses the proper dbxxx executable in each case, where both Darwin and LINUX executables exist. The linux executable has a CURC cross compiled executable compiled with gcc/10.2.0 for use in RC systems.

- Database output, header, and logs will be written in the correspondent ion sub-directory. Intermediary folders and files will be deleted upon completion. The logs are dynamically written, and calculation status can be checked anytime with *tail*; e.g.

        tail BASHJOB_0.LOG 




