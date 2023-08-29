import datetime as dt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GadiClient import GadiClient
from sklearn.cluster import KMeans as kmeans
from merge_data import get_env_clusters, last_day_of_month
from era5_spatial_cluster import transform_era5, era5_clustering
from merge_data_barpa import load_barpac_m_wg, load_barpac_m_lightning
import dask
import argparse

if __name__ == "__main__":

	#This script:
	#1) Loads BARPAC-M wind gust data, calculates the WGR at each point
	#2) Matches BARPA-R environmental diagnostics to those gusts, in a 100 x 100 km rolling maximum
	#3) Resamples the wind gusts to a daily maximum at each spatial point, and selects the WGR and environmental diagnostics corresponding to those maxima
	#4) Saves this spatial grid
	#5) For each day, calculates the domain maximum, and saves to a pandas DF
	#6) For each day, calculates the domain maximum for only daily maximum gusts that are "convective", and saves to a pandas DF

	client = GadiClient()
	dask.config.set({"array.slicing.split_large_chunks":True})

	parser = argparse.ArgumentParser()
	parser.add_argument('-start_year', type=int, help="Start year to crunch data. Will start in December of that year")
	parser.add_argument('-end_year', type=int, help="Year to stop. Will compute up to February of this year. Must be at least 1 year greater than start_year")
	parser.add_argument('-forcing', type=str, help="Can be era or cmip5")
	parser.add_argument('-experiment', type=str, help="Must be historical or rcp85")
	args = parser.parse_args()
	start_year = args.start_year
	end_year = args.end_year
	forcing = args.forcing
	experiment = args.experiment

	if forcing == "era":
		model = "erai"
		ensemble = "r0"
	elif forcing == "cmip5":
		model = "ACCESS1-0"
		ensemble = "r1i1p1"
	
	date1 = dt.datetime(int(start_year), 12, 1)
    
	while date1 <= dt.datetime(end_year,3,1):
		print(date1)
		date2 = last_day_of_month(date1)
		if (date1.month in [12,1,2]):
				fid1 = date1.strftime("%Y%m%d")
				fid2 = date2.strftime("%Y%m%d")
				fid = "0_"+fid1+"_"+fid2
				print(fid1)
		else:
				date1 = date2 + dt.timedelta(days=1)
				continue     

		#Load BARPAC-M data (10-minute wind gusts over whole domain)
		f,_,_=load_barpac_m_wg(fid,forcing,model,experiment,ensemble)

		#Now calculate the wind gust ratio.
		#Note assertion/assumption is that data is 10-minute, because xarray rolling can't handle time-aware windows
		assert f.time_coverage_resolution == 'PT9M59S', "CHECK TIME OUTPUT FREQUENCY, IS NOT EQUAL TO 10 MINS, SO ROLLING WINDOW MIGHT NEED TO BE CHANGED"
		f["wgr"] = f.max_wndgust10m / (f.max_wndgust10m.rolling(time=24, min_periods=6, center=True).mean()).persist()

		#Get the environmental information from BARPA-R. This includes variables for clustering
		if forcing == "cmip5":
			f_env = np.unique(["/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpa_access_"+pd.to_datetime(fid.split("_")[1]).strftime("%Y%m")+"*.nc",\
					"/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpa_access_"+(pd.to_datetime(fid.split("_")[2])+dt.timedelta(days=1)).strftime("%Y%m")+"*.nc"])
		elif forcing == "era":
			f_env = np.unique(["/g/data/eg3/ab4502/ExtremeWind/aus/barpa_erai/barpa_erai_"+pd.to_datetime(fid.split("_")[1]).strftime("%Y%m")+"*.nc",\
					"/g/data/eg3/ab4502/ExtremeWind/aus/barpa_erai/barpa_erai_"+(pd.to_datetime(fid.split("_")[2])+dt.timedelta(days=1)).strftime("%Y%m")+"*.nc"])
		env_ds = []
		for fname in f_env:
			env_ds.append(xr.open_mfdataset(fname,chunks={"time":-1})[["lr13","qmean01","Umean06","s06","bdsd"]])
		env = xr.concat(env_ds,dim="time")

		#Slice to times where we have wind gust output, using the closest 6-hourly time for each 10-minute time step
		f_times = pd.to_datetime(f.time.values).round("6H")
		env = env.sel(time=slice(f_times[0],f_times[-1]))

		#Resample all BARPA-R variables (clustering variables and BDSD) to a 100 x 100 km rolling max (8 x 8 grid points)
		dx_km = 12
		radius_km = 50
		radius_pixels = int(np.floor(radius_km*2 / dx_km))
		env = env.rolling(dim={"lat":radius_pixels, "lat":radius_pixels},center=True,min_periods=radius_pixels/2).max().persist()

		#Get the environmental clustering model based on Umean, lr13, qmean, and s06
		cluster_mod, cluster_input = get_env_clusters()

		#Perform the clustering on the BARPA-R data
		s06, qmean01, lr13, Umean06 = transform_era5(env, cluster_mod, cluster_input)
		s06 = s06.persist(); qmean01 = qmean01.persist(); lr13 = lr13.persist(); Umean06 = Umean06.persist()
		cluster = era5_clustering(s06, qmean01, lr13, Umean06, env, cluster_mod).persist()

		#Spatially interpolate the BARPA-R data to the BARPAC grid
		cluster["bdsd"] = cluster["bdsd"].interp({"lat":f.latitude,"lon":f.longitude},method="nearest").persist()
		cluster["cluster"] = cluster["cluster"].interp({"lat":f.latitude,"lon":f.longitude},method="nearest").persist()

		#Select 10-minute times from the environmental data (rounding to nearest 6-hourly time step)
		cluster = cluster.sel(time=f_times)
		cluster["time"] = f["time"]

		#Add environmental data to the gust dataset
		f["bdsd"] = cluster["bdsd"]
		f["cluster"] = cluster["cluster"]

		#Now calculate the daily maximum gust, and keep the wgr + environmental info for those gusts.
		#Best I could do is a loop for now. Think it should be efficient enough when qsubbed

		dmax_wgr = []
		dmax_bdsd = []
		dmax_cluster = []
		times = []
		for name, group in f.groupby("time.date"):
		    times.append(name)
		    dmax_wgr.append(xr.where(group.max_wndgust10m == group.max_wndgust10m.max("time"),group.wgr,np.nan).max("time"))
		    dmax_bdsd.append(xr.where(group.max_wndgust10m == group.max_wndgust10m.max("time"),group.bdsd,np.nan).max("time"))
		    dmax_cluster.append(xr.where(group.max_wndgust10m == group.max_wndgust10m.max("time"),group.cluster,np.nan).max("time"))    
		    
		dmax_wgr = xr.concat(dmax_wgr,"date").persist()
		dmax_wgr["date"] = times

		dmax_bdsd = xr.concat(dmax_bdsd,"date").persist()
		dmax_bdsd["date"] = times

		dmax_cluster = xr.concat(dmax_cluster,"date").persist()
		dmax_cluster["date"] = times

		#Resample gusts to daily max
		dmax_gust = f["max_wndgust10m"].groupby("time.date").max().persist()

		#Resample BDSD to daily max (using all time steps, rather than BDSD at time of max gust)
		dmax_bdsd_all = f["bdsd"].groupby("time.date").max().persist()

		#Load the daily max lightning field
		#l,_,_=load_barpac_m_lightning(fid,forcing,model,experiment,ensemble)
		#l["time"] = pd.to_datetime(l.time.values).floor("D")
		#l = l.rename({"time":"date"})

		#Combine into one Dataset
		#ds=xr.Dataset({"wgr_dmax":dmax_wgr,"gust_dmax":dmax_gust,"lightning":l["n_lightning_fl"],"bdsd":dmax_bdsd,"cluster":dmax_cluster})
		ds=xr.Dataset({"wgr_dmax":dmax_wgr,"gust_dmax":dmax_gust,"bdsd":dmax_bdsd,"cluster":dmax_cluster,"bdsd_dmax":dmax_bdsd_all})

		#Resample lightning strokes to a 100*100 km grid box
		#dx_km = 2.2
		#radius_km = 50
		#radius_pixels = int(np.floor(radius_km * 2 / dx_km))
		#ds["lightning100"] = ds.lightning.rolling(dim={"latitude":radius_pixels, "longitude":radius_pixels},center=True,min_periods=radius_pixels/2).sum()
		#if forcing == "cmip5":
		#	ds = ds.drop(["height","realization","forecast_period","forecast_reference_time","lightning"]).drop_indexes(["date","latitude","longitude"]).persist()
		#elif forcing == "era":
		#	ds = ds.drop(["height","forecast_period","forecast_reference_time","lightning"]).drop_indexes(["date","latitude","longitude"]).persist()

		#Mask ocean
		barpa_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/lnd_mask-BARPAC-M_km2p2.nc")
		ds = xr.where(barpa_lsm.lnd_mask==1,ds,np.nan)

		#We now have a daily max DataSet of gusts, with corresponding wgr, bdsd, and environmental cluster
		#Now we can output the daily-domain mas gust, using all gusts, lightning-associated gusts, and lightning-associated gusts with a wgr of at least 1.5
		ds["date"] = pd.to_datetime(ds.date.values)
		ds.to_netcdf("/scratch/eg3/ab4502/barpa_scw_"+model+"_"+experiment+"_"+fid+".nc",
		    encoding = {
			"wgr_dmax":{"zlib": True, "complevel": 9, "least_significant_digit":3},
			"gust_dmax":{"zlib": True, "complevel": 9, "least_significant_digit":3},
			"bdsd":{"zlib": True, "complevel": 9, "least_significant_digit":3},
			"cluster":{"dtype":int,"zlib": True, "complevel": 9, "_FillValue":-1}})
			#"lightning100":{"zlib": True, "complevel": 9, "least_significant_digit":0}})

		#Get the spatial locations of the domain maximum gust, and output as a pandas df
		argmax = ds.gust_dmax.argmax(["latitude","longitude"])
		ddmax = ds.isel(latitude=argmax["latitude"].expand_dims("points").compute(), 
				  longitude=argmax["longitude"].expand_dims("points").compute())
		ddmax_df = ddmax.to_dataframe()
		ddmax_df["date"] = ddmax["date"]

		#Do the same but for convective gusts
		argmax_conv = ds.gust_dmax.where(ds.wgr_dmax >= 1.5, -999).argmax(["latitude","longitude"],skipna=True)
		ddmax_conv = ds.isel(latitude=argmax_conv["latitude"].expand_dims("points").compute(), 
				   longitude=argmax_conv["longitude"].expand_dims("points").compute())
		ddmax_conv_df = ddmax_conv.to_dataframe()
		ddmax_conv_df["date"] = ddmax_conv["date"]

		#SAVE DATAFRAMES
		ddmax_df.to_csv("/scratch/eg3/ab4502/barpa_scw_ddmax_"+model+"_"+experiment+"_"+fid+".csv")
		ddmax_conv_df.to_csv("/scratch/eg3/ab4502/barpa_scw_ddmax_conv_"+model+"_"+experiment+"_"+fid+".csv")

		date1 = date2 + dt.timedelta(days=1)

	client.close()
