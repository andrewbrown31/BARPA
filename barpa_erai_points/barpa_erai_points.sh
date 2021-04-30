#!/bin/bash

#PBS -P eg3 
#PBS -q express
#PBS -l walltime=12:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_erai_points.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barpa_erai_points.e 
#PBS -lstorage=gdata/eg3+gdata/ma05+gdata/du7
 
#Set up conda/shell environments 
source activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/barpa_read.py 2005 2015

