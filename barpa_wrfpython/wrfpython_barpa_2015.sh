#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=48:00:00,mem=128GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_wrf_python_2015.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_wrf_python_2015.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/tp28
 
#Set up conda/shell environments 
source activate wrfpython3.6 

#Initialise date
d=2015-01-01
#Specify end date
while [ "$d" != 2020-01-01 ]; do

  start_time=$(date -d "$d" +%Y%m%d)"00"
  end_time=$(date -d "$d + 1 month - 1 day"  +%Y%m%d)"18"
  echo "INFO: RUNNING WRFPYTHON ON DATA FROM" $start_time "to" $end_time
  python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m barpa -r aus -t1 $start_time -t2 $end_time -e historical -barpa_experiment era --ens r0 --issave True --outname barpa_erai --params min

  #Advance date
  d=$(date -I -d "$d + 1 month")

done


