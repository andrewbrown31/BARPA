import xarray as xr
import numpy as np
import datetime as dt
from merge_data import last_day_of_month, calc_bdsd

if __name__ == "__main__":

	#We calculated convective diagnostics for BARPA-R in 2020.
	#However, there are a few diagnostics missing, that we want to include.
	#For the ERA-forced BARPA, we were able to re-run the original code to include these diagnostics (wrf_non_parallel.py).
	#However, for ACEESS-forced data, the pp26 stream has been deleted (sfc dewpoint, pressure, and wind gusts).
	#As a work-around, we have calculated only the new diagnostics we need, without those pp26 data (lr13, rhmin13, q_melting).

	#This script merges the original data created in 2020, with these new files containing lr13, rhmin13, and q_melting

	#These are saved to new files, for checking. After this the original 2020 files can be overwritten

	start_year = 1989
	end_year = 1990
	date1 = dt.datetime(int(start_year), 1, 1)

	while date1.year <= end_year:
		date2 = last_day_of_month(date1)
		if (date1.month in [12,1,2,3]) & (date1 < dt.datetime(end_year,4,1)):
			print(date1)            
			fid1 = date1.strftime("%Y%m%d")
			fid2 = date2.strftime("%Y%m%d")
			fid = fid1+"_"+fid2
			file_path1 = "/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpa_access_"+fid+".nc"
			file_path2 = "/g/data/eg3/ab4502/ExtremeWind/aus/barpa/barpa_access_append_"+fid+".nc"
			f1 = xr.open_dataset(file_path1)
			f2 = xr.open_dataset(file_path2)
		
			for v in list(f2.data_vars.keys()):
				f1[v] = f2[v]

			f1 = calc_bdsd(f1)                
                
			f1.to_netcdf("/scratch/eg3/ab4502/barpa_access_"+fid+".nc")
		date1 = date2 + dt.timedelta(days=1)
