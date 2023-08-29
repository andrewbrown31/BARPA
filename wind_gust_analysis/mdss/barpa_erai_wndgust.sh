#!/bin/bash
 
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l jobfs=2GB
#PBS -q copyq
#PBS -P eg3
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/eg3+massdata/du7
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/retrieve_mdss.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/retrieve_mdss.e

export MDSSVERBOSE=1

#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1990*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1991*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1992*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1993*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1994*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1995*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1996*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1997*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1998*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/1999*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/2000*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/2001*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/2002*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
#mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/era/erai/historical/r0/2003*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/era/erai/historical/r0/
