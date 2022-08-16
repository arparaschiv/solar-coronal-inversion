#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --partition=blanca-nso
#SBATCH --qos=blanca-nso
#SBATCH --job-name=CLEDB_database_invert
#SBATCH --output=CLEDBINV_JOBLOG.log

##  This script tests the 1-line setup for CLEDB inversion on slurm enabled systems.

### USAGE EXAMPLE (script runs directly from login node)
# module load slurm/blanca
# sbatch test_cledb_slurm.sh

  module purge

  source /curc/sw/anaconda3/latest
  conda activate CLEDBenv

  SCRATCH=${SLURM_SCRATCH}

  echo NodeName: $SLURMD_NODENAME
  AVAIL_SPACE=`df --output=avail $SCRATCH | grep -v Avail`

  if [ $AVAIL_SPACE -lt 1000000 ]                    ## 1 GB is just a check for the sake of checking; IO for this routine should be in the order of 10s of MB.
  then
      echo Not enough available temporary storage:
      echo Only ${AVAIL_SPACE} bytes available
      echo exiting
      exit 0
  else
      echo Available space: ${AVAIL_SPACE} bytes
      echo Continuing..
  fi

## nline controls the number of lines to be inverted. This just switched between the 1-line and 2-line setups which are procedurally different
nline=1

if (($nline -eq 1)); then
    python3 test_1line.py
  fi
if (($nline -neq 1)); then
    python3 test_2line.py
  fi
