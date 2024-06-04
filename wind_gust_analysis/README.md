# Log

This directory comprises a workflow as follows. This generates data used for manuscript figures, outlined below. 

The manuscript has been submitted to Natural Hazards and Earth System Science, and is titled [Convection-permitting climate model representation of severe convective wind gusts and future changes in southeastern Australia](https://doi.org/10.5194/egusphere-2024-322)

## Generate data
1) **Large-scale environmental diagnostics relavant for convection** are computed using data the regional BARPA configuration (BARPA-R) and the ERA5 reanalysis. The generic code for this computation is in [wrf_non_parallel.py](https://github.com/andrewbrown31/SCW-analysis/blob/master/wrf_non_parallel.py) within [this repository](https://github.com/andrewbrown31/SCW-analysis/tree/master), and is driven for BARPA by the code in the [`../barpa_wrfpython/`](https://github.com/andrewbrown31/BARPA/tree/main/barpa_wrfpython) and [`../barpa_access_wrfpython/`](https://github.com/andrewbrown31/BARPA/tree/main/barpa_access_wrfpython) directories. The two different BARPA directories represent runs forced by ERA-Interim and ACCESS1-0, respectively. For ERA5, the code is driven from a different repository ([`SCW-analysis/jobs/era5_wrfpython/`](https://github.com/andrewbrown31/SCW-analysis/tree/master/jobs/era5_wrfpython)). Note that we also used [`append_barpa_access_diagnostics.py`](append_barpa_access_diagnostics.py) to update some earlier BARPA diagnostic files.
2) **Severe convective wind event clusters are defined from observed events**, using [this code](https://github.com/andrewbrown31/tint_processing/blob/main/auto_case_driver/kmeans_and_cluster_eval.ipynb) presented in [this paper](https://doi.org/10.1175/MWR-D-22-0096.1).
3) **Retrieve some BARPA data from tape if needed** using the code in [`mdss/`](mdss).
4) **Extract wind gust, lightning, and large-scale environmental diagnostic data from BARPA** using [`merge_data_barpa.py`](merge_data_barpa.py). This includes an option to coarsen the 2.2 km BARPA configuration to 12 km. This code then merges the BARPA data with observations of wind gusts and lightning, as well as diagnostics and wind gusts from ERA5 and ERA-Interim reanalyses. Observations are resampled to 10-minute frequency (10-min max) to match the BARPA data. Environmental clustering from the previous step is also applied to BARPA and ERA5. All of the abovementioned variables are saved as `.csv` files with 10-minute intervals at a range of station locations in the BARPA SE Australia domain, from 2005-2015. [`merge_data_barpa.py`](merge_data_barpa.py) is run separately for each State in the domain (VIC, TAS, SA, NSW), to save on memory, noting that the observational wind gust files used here are separate for each State. This code is submitted to the PBS queue using `merge_data_barpa_*.sh` scripts
5) **Resample 10-minute data to daily maximum** from the above step, for more manageable data volumes for analysis, in the [`resample_daily_max.py`](resample_daily_max.py) script.
6) **Process BARPA data on the entire spatial grid**, rather than just for weather station locations as outlined above. This is intended for assessing spatial climatology and changes between historical and future climate forcing. The code for this data processing is in [`barpa_spatial_scw.py`](barpa_spatial_scw.py), and includes loading the BARPA data (wind gusts and large-scale diagnostics), doing clustering on the large-scale diagnostics, calculating the wind gust ratio for defining severe convective events, and saving daily maximum data. This code is submitted to the PBS queue using [`barpa_spatial_scw.sh`](barpa_spatial_scw.sh).
7) **Resample the spatial data to monthly** for plotting, while also counting the number of severe convective wind events in each cluster. This code is in [`spatial_barpa_analysis_prep.py`](spatial_barpa_analysis_prep.py) and submitted to the PBS job queue by [`spatial_barpa_analysis_prep.sh`](spatial_barpa_analysis_prep.sh).

## Analyse and plot

### Figure 1
[`plot_barpac_domains.ipynb`](plot_barpac_domains.ipynb)

### Figure 2, 3, 4, 5, 6 and 7, Table 1
[`analysis_10min_gusts.ipynb`](analysis_10min_gusts.ipynb)

### Figure 8
[`../lightning_analysis/lightning_compare_points.ipynb`](../lightning_analysis/lightning_compare_points.ipynb)

### Figure 9 and 10, Table 2 and 3
[`spatial_barpa_analysis.ipynb`](spatial_barpa_analysis.ipynb)

### Figure 11
[`future_distributions.ipynb](future_distributions.ipynb)
