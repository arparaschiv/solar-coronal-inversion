#!/bin/bash
##  SPLITS THE DATABASE GENERATION JOBS INTO AVAILABLE CPU THREADS

## catcher for kill signal so you cna ctrl+c/cmd+c the main script and kill all subjobs
trap "kill 0" SIGINT

## a wrapup for using architecture independent sed where "sed -i" has different implementations
## OSX uses BSD sed while linux uses GNU sed
## to run on OSX we require the install of gnu-sed (brew install gnu-sed) to activate the gsed command that mimics standard GNU syntax
sedi () {
    sed --version >/dev/null 2>&1 && sed -i -- "$@" || gsed -i "$@"
}


## a wrapper for selecting the correct executable for linux vs LINUX vs CURC;
## should not have any problem if os version changes as long as the "_linux", "_rclinux" and "_darwin" executables exist
case "$(uname -s)" in
    Linux*)     dbexec='db*_linux';;
    Darwin*)    dbexec='db*_darwin';;
    CYGWIN*)    dbexec='db*_linux';;
    MINGW*)     dbexec='db*_linux';;
    *)          exit 1
esac


## additional case for the CURC research computing systems running in interactive jobs (Only!). Separate script is provided for batch jobs
## This will make script scale correctly on RC systems and use the allocated resources in interactive jobs; e.g. slurm sinteractive spawned jobs
## this section needs access to slurm commands
RCsystem='rc.int.colorado.edu'                                             ## setup for the colorado university system; change here for your own;

uname -n | grep $RCsystem &> /dev/null                                     ## check if RC system in use
if [ $? == 0 ]; then
		dbexec='db*_rclinux';                                              ## a 3rd executable exists for this; compiled with gcc/10.2.0 available on the CURC Blanca system
		jobid=$(echo $(squeue -j -u $(whoami)) | cut -c 58-65);            ## get the job id of the running interactive job
		ncaval=$(scontrol show job $jobid | grep TRES | cut -c 39-);       ## number of processors/threads in system if running on research computing systems
	else
		ncaval=$(getconf _NPROCESSORS_ONLN);                               ## number of processors/threads in system if running on local machines
	fi
##.........................................................................## end slurm requiring section. (will not be used if $RCsystem does not match!)

re='^[0-9]+$'                                                              ## to check if ncore is a number variable
srt=$'Hi \nYou have '$ncaval$' CPU threads available. \nHow many to use?'
echo "$srt"
read -p 'Input:' ncore
if ! [[ "$ncore" =~ $re && "$ncore" -ge "1" && "$ncore" -le "$ncaval" ]]; then
	echo "Input not valid";
	exit 1;
fi

### Which line to generate database for
srt=$'Please indicate the line to use. Options are:\n1:    Fe XIII 1074.7nm\n2:    Fe XIII 1079.8nm\n3:    Si X    1430.1nm\n4:    Si IX   3934.3nm'
echo "$srt"
read -p 'Input:' linesel
if ! [[ "$linesel" =~ $re && "$linesel" -ge "1" && "$linesel" -le "4" ]]; then
	echo "Input not valid";
	exit 1;
fi

case "$linesel" in
    1*)     dbdir='fe-xiii_1074' inp='fe13a';;
    2*)     dbdir='fe-xiii_1079' inp='fe13b';;
    3*)     dbdir='si-x_1430'    inp='si10';;
    4*)     dbdir='si-ix_3934'   inp='si09';;
esac

### database and calculation configuration parameters
### please keep the DB.INPUT configuration as in example; including spaces and decimal numbers on line 5 for this to work
dbconf=$(sed -n 4p DB.INPUT)                                      ## read database parameter configuration
yh=$(echo $dbconf | sed -n 's/^[^0-9]*\([0-9]\{1,\}\).*$/\1/p')   ## requested number of Y heights
ln=$(sed -n 9p DB.INPUT)                                          ## parse parameter ranges line
ymin=$(echo "${ln[@]:16:5}")                                      ## requested minimum Y height
ymax=$(echo "${ln[@]:24:5}")                                      ## requested maximum Y height
dy=$(echo "scale=4 ; ($ymax - $ymin) / ($yh-1) " | bc)            ## delta for DB spacing resolution

if (($yh % 2 == 0)); then
	echo "Warning: even number of "$yh" height calculations selected"
	echo "Odd number of calculations will better match the requested YMAX"
fi

### MAIN LOOP; sends calculations to cpu threads
mkdir -p $dbdir                                                                                            ## go to the new database directory
cd $dbdir
rm -r ./*  2> /dev/null                                                                                    ## REMOVE ALL OLDER CALCULATIONS AND LOGS

if (($yh > $ncore)); then                                                                                  ## if generating more database heights than available threads
	subcalc=$(echo "scale=0 ; $yh / $ncore " | bc)                                                         ## minimum number of Y subcalculations for each CPU thread.
	subcalc1=$(echo "scale=0 ; $subcalc +1 " | bc)                                                         ## number of Y subcalculations +1 for each CPU thread.
	rescalc=$(echo "scale=0 ; $yh % $ncore " | bc)     
	                                                    ## residual calculations to add to the first created threads
	for e in `seq 0 $(echo "scale=0 ; $ncore - 1 " | bc)`; do                                              ## cycle through available threads
		mkdir $e;
		cd $e;
        \cp '../../'DB.INPUT ./;
		\cp '../../config/'{GRID.DAT,IONEQ} ./;                                                            ## make individual run dirs and copy necessary files
		\cp '../../config/'ATOM.${inp[@]:0:4} ./ATOM;                                                      ## substring trick because Fe XIII has two input files and 1 atom
		\cp '../../config/'INPUT.${inp[@]:0:5} ./INPUT;
		\cp '../../config/'$dbexec ./dbrun;                                                                ## executable is just dbrun

		##tasks will not evenly split between CPU threads; first rescalc number of threads will spawn subcalc+1 calculations
		if (($e < ($yh % $ncore ))); then                                                                  ## loop for first threads with 1 extra calculation each
			ymin_c=$(echo "scale=4; $ymin + ( $e      * $subcalc1 )* $dy       " | bc | awk '{printf("%.3f",$1)}');     ## min and max y height of each CPU thread subtask
			ymax_c=$(echo "scale=4; $ymin + (($e+1)   * $subcalc1 )* $dy - $dy " | bc | awk '{printf("%.3f",$1)}');
			dbconf_new="$subcalc1"${dbconf[@]:2}"";                                                        ## amended DB configuration to write to each thread (with the subcalc1 thread)
			#echo $ymin_c,$ymax_c,$dbconf_new
		else                                                                                               ## loop for remaining threads that do subcalc number of calculations
			ymin_c=$(echo "scale=4; $ymin + (( $e      * $subcalc ) +$rescalc )* $dy       " | bc | awk '{printf("%.3f",$1)}');  ## min and max y height of each CPU thread subtask
			ymax_c=$(echo "scale=4; $ymin + ((($e+1)   * $subcalc ) +$rescalc )* $dy - $dy " | bc | awk '{printf("%.3f",$1)}');  ## add rescalc to index heights correctly
			dbconf_new="$subcalc"${dbconf[@]:2}"";                                                         ## amended DB configuration to write to each thread
		    #echo $ymin_c,$ymax_c,$dbconf_new
		fi

		sedi '4d' DB.INPUT;                                                                                ## add computed number of subcalculations to DB.INPUT
		sedi "4i\\
		${dbconf_new} " DB.INPUT;

		sedi '9d' DB.INPUT;                                                                                ## add computed Y heights to DB.INPUT
		sedi "9i\\
		${ln[@]:1:15}$ymin_c${ln[@]:21:3}$ymax_c${ln[@]:29} " DB.INPUT;
		##two line versions of sedi are to make it POSIX compliant and work on OSX in linux it will add an extra tab to each line which will not matter when using

		## run the actula tasks, log the jobs in realtime, and save the processor IDS for each thread.
		./dbrun ${e} 2>&1 | tee ../BASHJOB_${e}.LOG tee 1> /dev/null &                                     ## tee is needed to write output to files in realtime!
		                                                                                                   ## tee 1 > /dev/null suppresses terminal output. 
		pids[${e}]=$!
		cd ..
	done
else                                                                                                       ## when there are less tasks than threads
	echo "Warning: Number of requested heights <= than number of threads!" 
	echo "Less than requested threads might be used!"
	subcalc=1                                                                                              ## subcalc is 1 here
	for e in `seq 0 $(echo "scale=0 ; $yh - 1 " | bc)`; do
		mkdir $e;
		cd $e;
        \cp '../../'DB.INPUT ./;
		\cp '../../config/'{GRID.DAT,IONEQ} ./;                                                            ## make individual run dirs and copy necessary files
		\cp '../../config/'ATOM.${inp[@]:0:4} ./ATOM;                                                      ## substring trick because Fe XIII has two input files and 1 atom
		\cp '../../config/'INPUT.${inp[@]:0:5} ./INPUT;
		\cp '../../config/'$dbexec ./dbrun;                                                                ## executable is just dbrun

		ymin_c=$(echo "scale=4; $ymin + ( $e      *$subcalc )* $dy       " | bc | awk '{printf("%.3f",$1)}');   ## min and max y height of each CPU thread subtask
		ymax_c=$(echo "scale=4; $ymin + (($e+1)   *$subcalc )* $dy - $dy " | bc | awk '{printf("%.3f",$1)}');
		dbconf_new="$subcalc"${dbconf[@]:2}"";                                                             ## amended DB configuration to write to each thread
		#echo $ymin_c,$ymax_c,$dbconf_new

		sedi '4d' DB.INPUT;                                                                                ## add computed number of subcalculations to DB.INPUT
		sedi "4i\\
		${dbconf_new} " DB.INPUT;

		sedi '9d' DB.INPUT;                                                                                ## add computed Y heights to DB.INPUT
		sedi "9i\\
		${ln[@]:1:15}$ymin_c${ln[@]:21:3}$ymax_c${ln[@]:29} " DB.INPUT;
		##two line versions of sedi are to make it POSIX compliant and work on OSX in linux it will add an extra tab to each line which will not matter when using

		## run the actula tasks, log the jobs in realtime, and save the processor IDS for each thread.
		./dbrun ${e} 2>&1 | tee ../BASHJOB_${e}.LOG tee 1> /dev/null &                                     ## tee is needed to write output to files in realtime!
		                                                                                                   ## tee 1 > /dev/null supresses terminal output. 
		pids[${e}]=$!
		cd ..
	done
fi

## waits for each CPU thread to finalize. 
## this is similar to C/fortran; script can not resolve order of tasks.e.g. runs are not sorted incrementaly speedwise.
## e.g. if the second PID finishes before the first, we only find out after the first PID finishes.
for pid in ${pids[*]}; do
    wait $pid
    echo "Process "$pid" completed!"
done

## after processes complete, append the CLEDB joblogs to the bash joblogs
## two branches based on the ratio between tasks and threads.
if (($yh >= $ncore)); then 
	for e in `seq 0 $(echo "scale=0 ; $ncore - 1 " | bc)`; do
		echo "CLEDB JOBLOG:" >> ./BASHJOB_${e}.LOG
		cat ./${e}/JOBLOG >> ./BASHJOB_${e}.LOG
	done
else
	for e in `seq 0 $(echo "scale=0 ; $yh - 1 " | bc)`; do
		echo "CLEDB JOBLOG:" >> ./BASHJOB_${e}.LOG
		cat ./${e}/JOBLOG >> ./BASHJOB_${e}.LOG
	done
fi

find . -name 'DB*.DAT' -exec mv -f {} ./ \;                                                            ## move the results from thread dirs to the atom dir.
\cp ./0/db.hdr ./db.hdr                                                                                ## save the database header
rm -r */                                                                                               ## remove the thread directories and other junk
cd ..
echo "All calculations are completed!"