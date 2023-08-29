#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=64GB
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/append_barpa_access_diagnostics.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/append_barpa_access_diagnostics.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37+gdata/rt52+scratch/tp28+gdata/ub4+scratch/eg3
#PBS -l jobfs=2GB

module use /g/data/hh5/public/modules
module load conda/analysis3

python working/BARPA/wind_gust_analysis/append_barpa_access_diagnostics.py
