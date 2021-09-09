#!/bin/bash
##  SPLITS THE DATABASE GENERATION JOBS INTO AVAILABLE CPU THREADS (SLURM version for BLANCA cluster)
## Execute:

# module load slurm/blanca
# sbatch rundb_1line_slurm.bash

#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --partition=blanca-nso
#SBATCH --qos=blanca-nso
#SBATCH --job-name=CLEDB_database_build
#SBATCH --output=CLEDB_JOBLOG.log

  module purge
  SCRATCH=${SLURM_SCRATCH}

  echo NodeName: $SLURMD_NODENAME
  AVAIL_SPACE=`df --output=avail $SCRATCH | grep -v Avail`

  if [ $AVAIL_SPACE -lt 30000000 ]                    ## 30 GB should be enough to generate 2-3 databases in detailed parameter spaces..
  then
      echo Not enough available temporary storage:
      echo Only ${AVAIL_SPACE} bytes available
      echo exiting
      exit 0
  else
      echo Available space: ${AVAIL_SPACE} bytes
      echo Continuing..
  fi


## catcher for kill signal so you cna ctrl+c/cmd+c the main script and kill all subjobs
trap "kill 0" SIGINT

## a wrapup for using architecture independent sed where "sed -i" has different implementations
## OSX uses BSD sed while linux uses GNU sed
## to run on OSX we require the install of gnu-sed (brew install gnu-sed) to activate the gsed command that mimics standard GNU syntax
sedi () {
    sed --version >/dev/null 2>&1 && sed -i -- "$@" || gsed -i "$@"
}

#### NOT NEEDED FOR HEADLESS RUNS; the number of cores is the number of tasks;
# ncaval=$(getconf _NPROCESSORS_ONLN)                               ## number of processors in system
# re='^[0-9]+$'                                                     ## to check if number
# srt=$'HI \nYou have '$ncaval$' CPU threads available. \nHow many to use?'
# echo "$srt"
# read -p 'Input:' ncore
# if ! [[ "$ncore" =~ $re && "$ncore" -ge "1" && "$ncore" -le "$ncaval" ]]; then
# 	echo "Input not valid";
# 	exit 1;
# fi
ncore=$ntasks ### ncore is the number of sbatch tasks

## MANUAL INPUT IS NOT A GOOD APPROACH FOR HEADLESS RUNS! linesel is the emission line for which to generate a database. 
## linesel is a choice between the 4 available options. (1=Fe XIII 1074nm; 2= Fe XIII 1079nm; 3=Si X 1430nm; 4= Si IX 3934nm)
# srt=$'Please indicate the line to use. Options are:\n1:    Fe XIII 1074.7nm\n2:    Fe XIII 1079.8nm\n3:    Si X    1430.1nm\n4:    Si IX   3934.3nm'
# echo "$srt"
# read -p 'Input:' linesel
# if ! [[ "$linesel" =~ $re && "$linesel" -ge "1" && "$linesel" -le "4" ]]; then
# 	echo "Input not valid";
# 	exit 1;
# fi

linesel=1

case "$linesel" in
    1*)     dbdir='fe-xiii_1074' inp='fe13a';;
    2*)     dbdir='fe-xiii_1079' inp='fe13b';;
    3*)     dbdir='si-x_1430'    inp='si10';;
    4*)     dbdir='si-ix_3934'   inp='si09';;
esac

## a wrapper for setting the correct executable for linux vs osx;
## should not have any problem if version change as long as the "_linux" and "_darwin" exist
case "$(uname -s)" in
    Linux*)     dbexec='db*_linux';;
    Darwin*)    dbexec='db*_darwin';;
    CYGWIN*)    dbexec='db*_linux';;
    MINGW*)     dbexec='db*_linux';;
    *)          exit 1
esac
## additional case for the CU research computing systems;a 3rd executable exists for this
uname -n | grep 'rc.int.colorado.edu' &> /dev/null
if [ $? == 0 ]; then dbexec='db*_rclinux'; fi


### database and calculation configuration parameters
### please keep the DB.INPUT configuration as in example; including spaces and decimal numbers on line 5 for this to work
dbconf=$(sed -n 3p config/DB.INPUT)                               ## read database parameter configuration
yh=$(echo $dbconf | sed -n 's/^[^0-9]*\([0-9]\{1,\}\).*$/\1/p')   ## requested number of Y heights
ln=$(sed -n 5p config/DB.INPUT)                                   ## parse parameter ranges line
ymin=$(echo "${ln[@]:16:5}")                                      ## requestet minimum Y height
ymax=$(echo "${ln[@]:24:5}")                                      ## requested maximum Y height

subcalc=$(echo "scale=0 ; $yh / ($ncore +1) " | bc)               ## number of Y subcalculations for each CPU thread (except last one).
dbconf_new="$subcalc"${dbconf[@]:2}""                             ## amended DB configuration to write to each thread


## main loop to send calculations to cpu threads
homedir=pwd                                                       ## record the store directory 
mkdir -p ${SCRATCH}/$dbdir
cd ${SCRATCH}/$dbdir
rm -r ./*  2> /dev/null                                                                                ## REMOVE ALL OLDER CALCULATIONS AND LOGS
for e in `seq 0 $ncore`; do
	mkdir $e;
	cd $e;
	\cp '$homedir/config/'{DB.INPUT,GRID.DAT,IONEQ} ./;                                                ## make individual run dirs and copy necesary files
	\cp '$homedir/config/'ATOM.${inp[@]:0:4} ./ATOM;                                                   ## substring trick because Fe XIII has two input files and 1 atom
	\cp '$homedir/config/'INPUT.${inp[@]:0:5} ./INPUT;
	\cp '$homedir/config/'$dbexec ./dbrun;                                                             ## executable is just dbrun

	##tasks will not evenly split between CPU threads; residual tasks will be given to last CPU threah
	if (($e < $ncore)); then                                                                           ## default tast split

		ymin_c=$(echo "scale=3; $ymin+0.001 + ( $e      *$subcalc)*(($ymax - $ymin) / $yh)  " | bc );  ## min and max y height of each CPU thread subtask
		ymax_c=$(echo "scale=3; $ymin+        (($e+1)   *$subcalc)*(($ymax - $ymin) / $yh)  " | bc );
	
	else                                                                                               ## extra tasks for last core
		ymin_c=$(echo "scale=3 ; $ymax_c + 0.001 " | bc);
		ymax_c=$(echo "scale=3 ; $ymax   + 0.001 " | bc);
		subcalc=$(echo "scale=0 ; $yh / ($ncore +1) + ($yh % ($ncore +1)) " | bc);                     ## update the number of tasks in this last case
		dbconf_new="$subcalc"${dbconf[@]:2}"";
	fi	

	sedi '3d' DB.INPUT;                                                                                ## add computed number of subcalculations to DB.INPUT
	sedi "3i\\
	${dbconf_new} " DB.INPUT;

	sedi '5d' DB.INPUT;                                                                                ## add computed Y heights to DB.INPUT
	sedi "5i\\
	${ln[@]:1:15}$ymin_c${ln[@]:21:3}$ymax_c${ln[@]:29} " DB.INPUT;
	##two line versions of sedi are to make it POSIX compliant and work on OSX in linux it will add an extra tab to each line which will not matter when using

	## run the actual tasks, log the jobs in realtime, and save the processor IDS for each thread.
	./dbrun ${e} 2>&1 | tee ../BASHJOB_${e}.LOG tee 1> /dev/null &                                     ## tee is needed to write output to files in realtime!
	                                                                                                   ## tee 1 > /dev/null suppresses terminal output. 
	pids[${e}]=$!
	cd ..
done

## waits for each CPU thread to finalize. 
## this is similar to C/fortran; script can not resolve order of tasks.e.g. runs are not sorted incrementally speedwise.
## e.g. if the second PID finishes before the first, we only find out after the first PID finishes.
for pid in ${pids[*]}; do
    wait $pid
    echo "process "$pid" completed!"
done

## after processes complete, append the CLEDB joblogs to the bash joblogs
for e in `seq 0 $ncore`; do
	echo "CLEDB JOBLOG:" >> ./BASHJOB_${e}.LOG
	cat ./${e}/JOBLOG >> ./BASHJOB_${e}.LOG
done
find . -name 'DB*.DAT' -exec mv -f {} ./ \;                                                            ## move the results from thread dirs to the atom dir.
\cp ./0/db.hdr ./db.hdr                                                                                ## save the database header
rm -r */                                                                                               ## remove the thread directories and other junk
cd ..

## If there is a dedicated project storage  we need to copy the finalized databases from #SCRATCH to there and then free up the scratch space.
## TBF later