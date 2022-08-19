README for running CLE database calculations on multiple CPU threads.
### [solar-coronal-inversion repository on github](https://github.com/arparaschiv/solar-coronal-inversion/)

Contact: Alin Paraschiv (arparaschiv@ucar.edu)

#### **History:**
- ARP: 20210617 - initial release 
- ARP: 20210827 - Added a slurm enabled version of the script for batch jobs on RC systems
- ARP: 20210915 - Rewrote the thread scaling to allocate tasks uniformly across threads; Both interactive and batch scripts now can utilize RC slurm capabilities. The interactive version can only use slurm allocated resources inside interactive jobs. The batch dedicated version can utilize scratch directories; It copies outputs in a user's project directory after finalizing tasks.


#### **SCOPE:**

This is a simple bash script implementation that launches separate parallel processes for the CLEDB_BUILD module.
Two versions are provisioned:

1. *[rundb_1line.sh](./rundb_1line.sh)*    (For local **interactive** runs; can be utilized inside slurm interactive environments too.)
2. *[rundb_1line_slurm.sh](./rundb_1line_slurm.sh)*    (For **batch** and/or headless runs.)

#### **INSTALL and USAGE:**

- make sure the scripts are executable:

        chmod u+x rundb_1line.sh
        chmod u+x rundb_1line_slurm.sh

- (Only on OSX) Install gnu-sed (See notes below):

        brew install gnu-sed

- (Optional if needed) OSX might have issues with running executables ("cannot execute binary file").
To fix try:

        xattr -d com.apple.quarantine /path/to/file

- (Optional) for interactive jobs on RC systems, the correct modules may need to be preloaded in order for scripts to execute. 

        module load slurm/blanca
        module load gcc/10.2.0                (gcc is preloaded automatically in the batch version of the script.)

- run interactive jobs with (after starting the interactive node; see README_SLURM):

        ./rundb_1line.sh

- run batch/headless jobs with:

        sbatch rundb_1line_slurm.sh

#### **NOTES:**

- The interactive *[rundb_1line.sh](./rundb_1line.sh)* script requires two manual keyboard user inputs.

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

- The batch *[rundb_1line_slurm.sh](./rundb_1line_slurm.sh)* script has no keyboard inputs, but has manually defined variables that control the ions to generate and system paths.

- Most directory and file pointers are dynamically linked to the CLEDB distribution directory. Local runs should run without interference
  Some directory/system containing variables are defined to be compatible with the CURC system (scratch, project, etc. dirs). These may need to be updated for different systems.

- ** NEWLY COMPLETED RUNS WILL DELETE/OVERWRITE PREVIOUSLY COMPUTED CALCULATIONS AND LOGS IN THE CORRESPONDENT SUBFOLDER**

- The scripts are configured to produce one line database outputs. All atomic data for the four ions of interest along with the configuration files 
  are available in the *config* directory. This setup selects the relevant inputs automatically.

- Outside of the two batch scripts, the only user editable file is the [config/DB.INPUT](./config/DB.INPUT) that configures the database number of calculations (parameter resolution).

- Database output, header, and logs will be written in the correspondent ion sub-directory. Intermediary folders and files will be deleted upon completion.
  The logs are dynamically written and calculation status can be checked anytime with *tail*; e.g.

        tail BASHJOB_0.LOG 

- The ./rundb scripts will wait for all thread tasks to finish before exiting.
  Due to limitation in CPU process ID (PID) tracking, the user is not notified in order of threads finalizing, but in the order they were scheduled.
  e.g. if thread 2 finishes before thread 0, the user will find out only after thread 0 and thread 1 finish.
  A bug might manifest if a new unrelated task is scheduled with the same PID as one of the runs, but this should not occur in normal circumstances.
  If such a case occurs, a tail of the logs will verify that everything went well and scripts can be exited manually.

- The number of Y heights to calculate between the ymin and ymax ranges are not always a multiple of the number of CPU threads.
  The scripts will efficiently scale the tasks on the available threads.
  If less tasks (via [DB.INPUT](./config/DB.INPUT)) than threads (via keyboard or sbatch) are requested, the script will not utilize all allocated resources.

- The script heavily relies on the SED function. 
  SED has different implementations on Linux (GNU) vs mac (BSD) which make commands not be directly correspondent.
  A function wrapper *SEDI* that disentangles GNU vs BSD syntax is provided in the scripts.
  OSX users need to install a gnu implementation of sed (gnu-sed) for the script to be portable between systems (via the gsed command).

        brew install gnu-sed

- The script cuts and appends midline on the DB.INPUT file, to set the ymin and ymax ranges for each CPU thread.
  The number of decimals for all variables and 3 spaces in between them need to be kept in the configuration file in order to not introduce bugs.

- Executables (dbxxx) need to be build (from CLE) on the current architecture: ELF(linux) or Mach-O(OSX) 
  If non-correct executables are called a "cannot execute binary file" error is produced. Architecture can be checked with the file command 
  The configuration now deduces the OS in use and selects and uses the proper dbxxx executable in each case, where both Darwin and LINUX executables exist.
  The linux executable has a CURC cross compiled executable compiled with gcc/10.2.0 for use in RC systems.




