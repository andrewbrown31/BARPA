#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=24:00:00,mem=64GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/spatial_barpa_analysis_prep.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/spatial_barpa_analysis_prep.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37+gdata/rt52+scratch/tp28+gdata/ub4+scratch/eg3
#PBS -l jobfs=16GB

module use /g/data/hh5/public/modules
module load conda/analysis3

python working/BARPA/wind_gust_analysis/spatial_barpa_analysis_prep.py
