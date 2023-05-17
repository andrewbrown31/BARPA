#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=01:00:00,mem=32GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/resample_daily_max.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/resample_daily_max.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37+gdata/rt52+scratch/tp28+gdata/ub4

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

python working/BARPA/wind_gust_analysis/resample_daily_max.py
