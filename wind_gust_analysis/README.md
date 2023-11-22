# Log

This directory comprises a workflow as follows. This generates data used for manuscript figures, outlined below. The manuscript focuses on severe convective wind representation in the BARPA model, and future climate projections based on a convection-allowing configuration of the model, compared with a regional configuration.

## Generate data
1) **Large-scale environmental diagnostics relavant for convection** are computed from the regional BARPA configuration (BARPA-R) and the ERA5 reanalysis, using the [wrf_non_parallel.py](https://github.com/andrewbrown31/SCW-analysis/blob/master/wrf_non_parallel.py) script within [this repository](https://github.com/andrewbrown31/SCW-analysis/tree/master). This script is driven for BARPA by the code in the `../barpa_wrfpython/` and `../barpa_access_wrfpython/` directories, and for ERA5 in a different repository ([`SCW-analysis/jobs/era5_wrfpython/`](https://github.com/andrewbrown31/SCW-analysis/tree/master/jobs/era5_wrfpython)).
2) **Clustering into different types of severe convective wind events** (https://github.com/andrewbrown31/tint_processing/blob/main/auto_case_driver/kmeans_and_cluster_eval.ipynb)
3) **Extract wind gust, lightning, and large-scale environmental diagnostic data from BARPA**, and merge with observations of wind gusts and lightning, as well as data from ERA5 and ERA-Interim reanalyses (https://github.com/andrewbrown31/BARPA/blob/main/wind_gust_analysis/merge_data_barpa.py). Save as .csv. 10-minute frequency at station locations
4) **Resample 10-minute data to daily maximum** (resample_daily_max.py)

## Analyse and plot

### Figure 1
(analysis_10min_gust.ipynb)
