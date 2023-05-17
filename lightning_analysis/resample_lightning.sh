#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=01:00:00,mem=32GB 
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/resample_lightning.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/resample_lightning.e
#PBS -l storage=gdata/eg3+gdata/tp28+gdata/hh5+gdata/cj37+scratch/tp28

module use /g/data/hh5/public/modules
module load conda/analysis3

python working/BARPA/lightning_analysis/resample_lightning.py

