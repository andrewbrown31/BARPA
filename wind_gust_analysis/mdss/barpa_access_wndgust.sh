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

mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/20391130T0000Z/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2040*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2041*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2042*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2043*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2044*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2045*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2046*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2047*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2048*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2049*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2050*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2051*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2052*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2053*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2054*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2055*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2056*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2057*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
mdss -P du7 get "barpa/trials/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/2058*/*/pp26/max_wndgust10m*.nc" /scratch/eg3/ab4502/BARPAC-M_km2p2/cmip5/ACCESS1-0/rcp85/r1i1p1/
