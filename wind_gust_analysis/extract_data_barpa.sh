#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=190GB 
#PBS -l ncpus=8
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/extract_data_barpa.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/extract_data_barpa.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37+gdata/rt52+scratch/tp28+gdata/ub4

module use /g/data/hh5/public/modules
module load conda/analysis3

python working/BARPA/wind_gust_analysis/extract_data_barpa.py -state sa
python working/BARPA/wind_gust_analysis/extract_data_barpa.py -state vic
python working/BARPA/wind_gust_analysis/extract_data_barpa.py -state nsw
python working/BARPA/wind_gust_analysis/extract_data_barpa.py -state tas

