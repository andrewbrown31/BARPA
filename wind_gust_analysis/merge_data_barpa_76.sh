#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=32GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/merge_data_barpa_76.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/merge_data_barpa_76.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37

module use /g/data/hh5/public/modules
module load conda/analysis3

python working/BARPA/wind_gust_analysis/merge_data_barpa.py -start_year 2012 -rid 76 -state tas
