#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=12:00:00,mem=64GB 
#PBS -l ncpus=8
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/merge_data_barpa_vic.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/merge_data_barpa_vic.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37+gdata/rt52+scratch/tp28+gdata/ub4

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

python working/BARPA/wind_gust_analysis/merge_data_barpa.py -start_year 2005 -rid 0 -state vic
