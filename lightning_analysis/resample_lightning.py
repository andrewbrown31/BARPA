import xarray as xr
import datetime as dt
import numpy as np
import glob
from barpa_read import drop_duplicates
import pandas as pd
from dask.diagnostics import ProgressBar

def preprocess(ds):
    year = ds["time"].units.split(" ")[-1]
    ds = ds.assign_coords(time=[dt.datetime(int(year),1,1,0) + dt.timedelta(hours=int(i)) for i in np.arange(len(ds.time.values))])
    return ds

if __name__ == "__main__":

	#Get all the file names for the BARPA lightning data, from 2005-2015    
	files = []
	start_year=2005; end_year=2015
	for y in np.arange(start_year,end_year+1):
		for m in ["12","01","02"]:
			fid = "x_" + str(y) + m
			barpa_str1 = fid.split("_")[1]
			barpa_str2 = (dt.datetime.strptime(fid.split("_")[1]+"01","%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m")

			files = files + glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+\
						  barpa_str1+"*/pp0/*.nc")
			files = files + glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+\
						  barpa_str2+"*/pp0/*.nc")
	
	#Load the BARPA lightning data    
	barpac_m = drop_duplicates(xr.open_mfdataset(np.unique(files), concat_dim="time", combine="nested"))
	barpac_m = barpac_m.sel({"time":(np.in1d(barpac_m["time.year"], np.arange(start_year,end_year+1))) & \
                        (np.in1d(barpac_m["time.month"],[12,1,2]))})

	#Load the WWLLN data, and resample to daily intervals by summing strokes over the hourly intervals
	wwlln = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_1hr/*.nc", decode_times=False, preprocess=preprocess)
	wwlln = wwlln.resample({"time":"1D"}).sum()
	wwlln = wwlln.sel({"time":(np.in1d(wwlln["time.year"], np.arange(start_year,end_year+1))) & \
				(np.in1d(wwlln["time.month"],[12,1,2])) & \
			  (pd.to_datetime(wwlln.time.values) <= dt.datetime(2015,3,1))})

	#Subset both datasets to the same spatial bounds
	wwlln_subset = wwlln.sel({"lat":slice(barpac_m.latitude.min(), barpac_m.latitude.max()),
		 "lon":slice(barpac_m.longitude.min(), barpac_m.longitude.max())})
	barpa_subset = barpac_m.sel({"latitude":slice(wwlln.lat.min(), wwlln.lat.max()),
		 "longitude":slice(wwlln.lon.min(), wwlln.lon.max())})

	#Bin the BARPA lightning data onto the WWLLN grid, and find the number of total days with at least one lightning flash in each grid box
	#Bins are defined with 0.25 degree lat-lon spacing, same as WWLLN, with WWLLN lat-lon coordinates in the centre of the box
	dlon = dlat = 0.25
	lon_bins = np.array(list(wwlln_subset.lon.values) + [wwlln_subset.lon.values.max()+dlon]) - (dlon/2)
	lat_bins = np.array(list(wwlln_subset.lat.values) + [wwlln_subset.lat.values.max()+dlat]) - (dlat/2)
	lon_labels = wwlln_subset.lon.values
	lat_labels = wwlln_subset.lat.values
	barpa_binned = (barpa_subset.n_lightning_fl.\
				groupby_bins("longitude",lon_bins,labels=lon_labels).sum().\
				groupby_bins("latitude",lat_bins,labels=lat_labels).sum() >= 1).sum("time")

	#For the daily WWLLN data, again find the number of total days with at least one lighting flash in each grid box
	wwlln_binned = (wwlln_subset.Lightning_observed >= 1).sum("time")

	#Update attributes for both datasets to retain original metadata
	barpa_binned = barpa_binned.assign_attrs({\
					"original_times":[dt.datetime.strftime(t,"%Y-%m-%d") for t in pd.to_datetime(barpa_subset.time.values)],\
					"original_lat":barpa_subset.latitude.values,\
					"original_lon":barpa_subset.longitude.values})
	wwlln_binned = wwlln_binned.assign_attrs({\
					"original_times":[dt.datetime.strftime(t,"%Y-%m-%d") for t in pd.to_datetime(wwlln_subset.time.values)]})

	#Save both datasets as netcdf
	ProgressBar().register()
	barpa_binned.to_netcdf("/g/data/eg3/ab4502/regridded_lightning/barpa.nc")
	wwlln_binned.to_netcdf("/g/data/eg3/ab4502/regridded_lightning/wwlln.nc")
