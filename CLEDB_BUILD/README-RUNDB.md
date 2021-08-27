README for running CLE database calculations on multiple CPU threads.

Contact: Alin Paraschiv (arparaschiv@nso.edu)

History:
- ARP: 20210617 - initial release
- ARP: 20210827 - Added a slurm enabled version of the script for RC systems

SCOPE:

This is a simple bash implementation that launches separate parallel processes.

INSTALL and USAGE:

- make sure the scripts are executable  
chmod u+x rundb_1line.sh
chmod u+x rundb_1line_slurm.sh

- (Optional if needed) OSX might have issues with running executables ("cannot execute binary file").
To fix try:
xattr -d com.apple.quarantine /path/to/file

- (Only on OSX) Install gnu-sed (See notes below)
  brew install gnu-sed

- run with:

./rundb_1line

- The only user editable file is the DB.INPUT that configures the database number of calculations (resolution).

NOTES:
- Two manual keyboard user inputs are required. 
    i. select how many CPU threads to use; 

        "HI 
        You have xx CPU threads available. 
        How many to use?""

    ii. which ion/line to compute. Each ion/line will create its own subfolder in the directory structure to store computations.

        "Please indicate the line to generate. Options are:
        1:    FE XIII 1074.7nm
        2:    FE XIII 1079.8nm
        3:    Si X    1430.1nm
        4:    Si IX   3934.3nm"

    NEWLY COMPLETED RUNS WILL DELETE/OVERWRITE PREVIOUSLY COMPUTED CALCULATIONS AND LOGS IN THE CORRESPONDENT SUBFOLDER


- The script is configured to produce one line database outputs.All atomic data for the four ions of interest along with the configuration files 
  are directly available and this setup selects the relevant inputs automatically.

- Database output, header, and logs will be written in the correspondent subdirectory. Intermediary folders and files will be deleted upon completion.
  The logs are dynamically written and calculation status can be checked anytime with  commands like tail; e.g.

tail BASHJOB_0.LOG 

- The ./run scripts will wait for all thread tasks to finish before exiting.
  Due to limitation in CPU process ID (PID) tracking, the user is not notified in order of threads finalizing, but in the order they were scheduled.
  e.g. if thread 2 finishes before thread 0, the user will find out only after thread 0 and thread 1 finish.
  A bug might manifest if a new unrelated task is scheduled with the same PID as one of the runs, but this should not occur in normal circumstances.
  If such a case occurs, a tail of the logs will verify that everything went well and scripts can be exited.

- The number of Y heights to calculate between the ymin and ymax ranges are not always a multiple of the number of CPU threads.
  Because of this, the last CPU thread might append a larger number of calculations in certain situations and be slower.

- The script heavily relies on the SED function. 
  SED has different implementations on Linux (GNU) vs mac (BSD) which make commands not be directly correspondent.
  A function wrapper SEDI that disentangles GNU vs BSD syntax is provided in the scripts.
  OSX users need to install a gnu implementation of sed (gnu-sed) for the script to be portable between systems (via the gsed command)

brew install gnu-sed

- The script cuts and appends midline on the DB.INPUT file, when setting the ymin and ymax ranges for each CPU thread.
  The number of decimals for all variables and 3 spaces in between them need to be kept in the configuration file in order to not introduce bugs.

- Executables (dbxxx) need to be build (from CLE) on the current architecture: ELF(linux) or Mach-O(OSX) 
  If non-correct executables are called a "cannot execute binary file" error is produced. Architecture can be checked with the file command 
  The configuration now deduces the OS in use and selects and uses the proper dbxxx executable in each case, where both Darwin and LINUX executables exist.





