import os
from sklearn.cluster import KMeans as kmeans
import argparse
import joblib
from GadiClient import GadiClient
import xarray as xr
import glob
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import tqdm
import netCDF4 as nc
from merge_data import last_day_of_month, latlon_dist, load_stn_info, subset_era5, extract_era5_df, get_env_clusters
from GadiClient import GadiClient

def load_barra_env_wg10(fid):
	barra_str = fid.split("_")[1]+"_"+fid.split("_")[2]
	barra = xr.open_mfdataset("/g/data/cj37/BARRA/BARRA_R/v1/analysis/spec/wndgust10m/"+\
                fid.split("_")[1][0:4] + "/" + fid.split("_")[1][4:6] + "/wndgust10m*" + fid.split("_")[1][0:6] + "*.nc",\
		    combine="nested",concat_dim="time").sortby("time")
	barra["wndgust10m"] = barra.wndgust10m.shift({"time":1}).pad(mode="edge")
	barra = barra.resample({"time":"1D"}).max()
	barra_lat = barra["latitude"].values
	barra_lon = barra["longitude"].values
	x,y = np.meshgrid(barra_lon,barra_lat)
	return barra, x, y

def load_barpa_r_wg(fid):

	barpa_str1 = fid.split("_")[1]
	barpa_str2 = (dt.datetime.strptime(fid.split("_")[1],"%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")

	files1 = glob.glob("/g/data/tp28//ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str1[0:6]+"*/pp26/max_wndgust10m-pp26-BARPAC-M_km2p2-*.nc")
	files2 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str2[0:6]+"*/pp26/max_wndgust10m-pp26-BARPAC-M_km2p2-*.nc")

	barpa = xr.open_mfdataset(files2+files1).sel({"time":slice(dt.datetime.strptime(fid.split("_")[1],"%Y%m%d"),dt.datetime.strptime(fid.split("_")[2],"%Y%m%d"))})

	barpa_lat = barpa["latitude"].values
	barpa_lon = barpa["longitude"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, x, y

def load_barpa_env(fid):
	barpa_str = fid.split("_")[1]+"_"+fid.split("_")[2]
	barpa = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_erai/barpa_erai_"+barpa_str+".nc", chunks={"time":16})
	barpa["wg10"] = barpa.wg10.shift({"time":1}).pad(mode="edge")
	barpa_dmax = barpa.resample({"time":"1D"}).max()
	barpa_lat = barpa["lat"].values
	barpa_lon = barpa["lon"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, barpa_dmax, x, y

def load_barpac_m_wg(fid):
    
	#Load 10-minute BARPAC-M data from /scratch. Also resample to hourly maximum.
    
	barpa_str1 = fid.split("_")[1]
	barpa_str2 = (dt.datetime.strptime(fid.split("_")[1],"%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")

	files1 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str1[0:6]+"*/pp26/max_wndgust10m-pp26-BARPAC-M_km2p2-*.nc")
	files2 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str2[0:6]+"*/pp26/max_wndgust10m-pp26-BARPAC-M_km2p2-*.nc")

	barpa = xr.open_mfdataset(files2+files1).sel({"time":slice(dt.datetime.strptime(fid.split("_")[1],"%Y%m%d"),dt.datetime.strptime(fid.split("_")[2],"%Y%m%d"))})

	barpa_lat = barpa["latitude"].values
	barpa_lon = barpa["longitude"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, x, y

def load_barpac_m_wg_daily(fid):
    
	#Load daily gust data from BARPAC-M
    
	barpa_str = fid.split("_")[1]
	barpa = xr.open_dataset(glob.glob("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/era/erai/historical/r0/pp_unified/daily/sfcWindGustmax/0p02deg/"+\
		barpa_str[0:4]+"/sfcWindGustmax_BARPAC-M_km2p2_erai_historical_r0_BARPAC_daily_"+barpa_str+"*.nc")[0])
	barpa_lat = barpa["lat"].values
	barpa_lon = barpa["lon"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, x, y

def get_points(da, lsm, stn_lat, stn_lon, stn_id):

	#Rather than getting the maximum/mean/min in a radius, this function gets the closest land point.
    
        lat = da.lat.values
        lon = da.lon.values
        x,y = np.meshgrid(lon,lat)
        x[lsm==0] = np.nan
        y[lsm==0] = np.nan
        
        dist_lon = []
        dist_lat = []
        for i in np.arange(len(stn_lat)):

                dist = np.sqrt(np.square(x-stn_lon[i]) + \
                        np.square(y-stn_lat[i]))
                temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
                dist_lon.append(temp_lon)
                dist_lat.append(temp_lat) 
                
        temp_df = da.isel(lat = xr.DataArray(dist_lat, dims="points"), \
                        lon = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()
        temp_df = temp_df.reset_index()  
        
        for p in np.arange(len(stn_id)):
                temp_df.loc[temp_df.points==p,"stn_id"] = stn_id[p]        
        
        temp_df = temp_df.drop(["points"],axis=1)        
        
        return temp_df

def load_tint_aws_barpa(fid, state, summary="max"):

	#########################################################################
	#   Observations							#
	#########################################################################

	if os.path.exists("/g/data/eg3/ab4502/TINTobjects/"+fid+"_aws.csv"):
		#Load the AWS one-minute gust data with TINT storm object ID within 10 and 20 km
		tint_df = pd.read_csv("/g/data/eg3/ab4502/TINTobjects/"+fid+"_aws.csv")
		tint_df["dt_utc"] = pd.DatetimeIndex(tint_df.dt_utc)
		tint_df["day"] = tint_df.dt_utc.dt.floor("D")
		stn_list = np.sort(tint_df.stn_id.unique())
	else:
		print(fid + "_aws.csv does not extist")
		return

	#Calculate the 4-hour wind gust ratio, for our SCW definition
	rolling4 = \
	    tint_df[["gust","stn_id","dt_utc"]].set_index(tint_df.dt_utc).groupby("stn_id").gust.rolling("4H",center=True,closed="both",min_periods=60).mean()
	rolling4.name = "rolling4"
	tint_df = pd.merge(tint_df,rolling4,on=["stn_id","dt_utc"])
	tint_df["wgr_4"] = tint_df["gust"] / tint_df["rolling4"]

	#Resample the one-minute AWS dataframe to daily maxima.
	#This is done by sorting the dataframe, such that
	#   1) SCWs are retained if they occur on a given day
	#   2) After that, the strongest gust is retained
	tint_df["scw"] = (tint_df["wgr_4"] >= 2) & (tint_df["in10km"]) & (tint_df["gust"]>=25) * 1
	dmax = tint_df.sort_values(["stn_id","day","scw","gust"])[["stn_id","day","dt_utc","gust","in10km","wgr_4","scw"]].groupby(["stn_id","day"]).last()
	dmax["dt_floor_6H"] = dmax.dt_utc.dt.floor("6H")
	dmax["dt_floor_1H"] = dmax.dt_utc.dt.floor("1H")

	#Load AWS station information
	stn_info = load_stn_info(state)
	stn_latlon = stn_info.set_index("stn_no").loc[stn_list][["lat","lon"]].values
	stn_lat = stn_latlon[:,0]; stn_lon = stn_latlon[:,1]

	#########################################################################
	#   BARPA (12 km)							#
	#########################################################################

	#Load BARPA 12 km environmental diagnostics
	barpa_env, barpa_dmax_env, x_env, y_env = load_barpa_env(fid)

	#Subset to within 50 km
	barpa_subset_env, rad_lats_env, rad_lons_env = subset_era5(barpa_env, stn_lat, stn_lon, y_env, x_env)
	barpa_dmax_subset_env, rad_lats_env, rad_lons_env = subset_era5(barpa_dmax_env, stn_lat, stn_lon, y_env, x_env)

	#Mask ocean
	barpa_env_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").\
			    interp({"latitude":barpa_subset_env.lat, "longitude":barpa_subset_env.lon}, method="nearest").lnd_mask.values
	barpa_subset_env = xr.where(barpa_env_lsm, barpa_subset_env, np.nan).persist()
	barpa_dmax_subset_env = xr.where(barpa_env_lsm, barpa_dmax_subset_env, np.nan).persist()

	#Get dataframe from the BARPA data
	barpa_df_env = extract_era5_df(barpa_subset_env, rad_lats_env, rad_lons_env, stn_list,"max")
	barpa_dmax_df_env = extract_era5_df(barpa_dmax_subset_env, rad_lats_env, rad_lons_env, stn_list,"max")

	#Get point data for the model wind gust (no radial max), and merge
	barpa_df_env_point = get_points(barpa_subset_env.wg10, barpa_env_lsm, stn_lat, stn_lon, stn_list).rename(columns={"wg10":"wg10_12km_point"})
	barpa_dmax_df_env_point = get_points(barpa_dmax_subset_env.wg10, barpa_env_lsm, stn_lat, stn_lon, stn_list).rename(columns={"wg10":"wg10_12km_point_dmax"})
	#barpa_df_env_point["time"] = barpa_df_env_point.time.dt.floor("1D")
	barpa_df_env = pd.merge(barpa_df_env.rename(columns={"wg10":"wg10_12km_rad"}), barpa_df_env_point, on=["time","stn_id"])
	barpa_dmax_df_env = pd.merge(barpa_dmax_df_env.rename(columns={"wg10":"wg10_12km_rad_dmax"}), barpa_dmax_df_env_point, on=["time","stn_id"])
		#		[["time","wg10_12km_rad_dmax","stn_id","lat","lon","wg10_12km_point_dmax"]]

	#Merge the daily and sub-daily data
	barpa_df_env["day"] = barpa_df_env["time"].dt.floor("1D")
	barpa_df_env = pd.merge(\
			barpa_dmax_df_env.drop(columns=["lat","lon"]),\
			barpa_df_env.drop(columns=["lat","lon"]),\
		    left_on=["time","stn_id"], right_on=["day","stn_id"],suffixes=["_dmax",""]).rename(columns={"time":"barpa_env_time"}).drop(columns=["time_dmax"])

	#Load clustering classification model saved by ~/working/observations/tint_processing/auto_case_driver/kmeans_and_cluster_eval.ipynb
	#TODO: this code works but currently just uses the daily maximum data. Ideally need to do this sub-daily
	#Order = [1,2,0] for clusters = [strong bg wind, steep lapse rate, high moisture]
	cluster_mod, cluster_input = get_env_clusters()
	input_df = (barpa_df_env[["s06","qmean01","lr13","Umean06"]]\
		   - cluster_input.min(axis=0))\
	    / (cluster_input.max(axis=0) - \
	       cluster_input.min(axis=0))
	barpa_df_env["cluster"] = cluster_mod.predict(input_df)
	input_df = (barpa_df_env[["s06_dmax","qmean01_dmax","lr13_dmax","Umean06_dmax"]]\
		   - cluster_input.min(axis=0))\
	    / (cluster_input.max(axis=0) - \
	       cluster_input.min(axis=0))
	barpa_df_env["cluster_dmax"] = cluster_mod.predict(input_df)

	#########################################################################
	#   BARPA (2.2 km)							#
	#########################################################################

	#Put in a check for BARPA data
	isbarpa_m = len(glob.glob("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/era/erai/historical/r0/pp_unified/daily/sfcWindGustmax/0p02deg/"+\
                fid.split("_")[1][0:4]+"/sfcWindGustmax_BARPAC-M_km2p2_erai_historical_r0_BARPAC_daily_"+fid.split("_")[1]+"*.nc")) > 0
	if isbarpa_m:

		#Load the BARPA 2.2 km wind gusts
		barpa_wg, x_wg, y_wg = load_barpa_wg(fid)

		#Subset to within r km
		barpa_subset_wg, rad_lats_wg, rad_lons_wg = subset_era5(barpa_wg, stn_lat, stn_lon, y_wg, x_wg, r=10)

		#Mask ocean
		barpa_wg_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/lnd_mask-BARPAC-M_km2p2.nc").\
				    interp({"latitude":barpa_subset_wg.lat, "longitude":barpa_subset_wg.lon}, method="nearest").lnd_mask.values
		barpa_subset_wg = xr.where(barpa_wg_lsm, barpa_subset_wg.sfcWindGustmax, np.nan).persist()

		#Get dataframe from the BARPA data
		barpa_df_wg = extract_era5_df(barpa_subset_wg, rad_lats_wg, rad_lons_wg, stn_list,"max")
		barpa_df_wg["time"] = barpa_df_wg.time.dt.floor("1D")

		#Get dataframe using the closest point function. Merge them with the other dataframes
		barpa_df_wg_point = get_points(barpa_subset_wg, barpa_wg_lsm, stn_lat, stn_lon, stn_list).rename(columns={"sfcWindGustmax":"wg10_2p2_point"}).\
					    drop(columns="height")
		barpa_df_wg_point["time"] = barpa_df_wg_point.time.dt.floor("1D")
		barpa_df_wg = pd.merge(barpa_df_wg.rename(columns={"sfcWindGustmax":"wg10_2p2_rad"}), barpa_df_wg_point, on=["time","stn_id"])

	#########################################################################
	#   BARRA (12 km)							#
	#########################################################################

	barra_wg, x_barra, y_barra = load_barra_env_wg10(fid)
	barra_wg = barra_wg.rename({"latitude":"lat","longitude":"lon"})
	barra_lsm = xr.open_dataset("/g/data/cj37/BARRA/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").lnd_mask

	#Subset to within 50 km
	barra_subset_wg, rad_lats_env, rad_lons_env = subset_era5(barra_wg, stn_lat, stn_lon, y_barra, x_barra)
	barra_subset_wg = xr.where(barra_lsm.sel({"longitude":barra_subset_wg.lon,"latitude":barra_subset_wg.lat}), barra_subset_wg, np.nan).persist()
	barra_df_wg = extract_era5_df(barra_subset_wg, rad_lats_env, rad_lons_env, stn_list,"max")
	
	barra_df_wg10_point = get_points(barra_wg.wndgust10m,\
		    barra_lsm.values, stn_lat, stn_lon, stn_list).rename(columns={"wndgust10m":"wg10_barra_12km_point"}).\
			drop(columns=["forecast_period","height"])

	barra_df_wg = pd.merge(barra_df_wg.rename(columns={"wndgust10m":"wg10_barra_12km_rad"}), barra_df_wg10_point)

	#########################################################################
	#   Merge together							#
	#########################################################################

	#Merge the two barpa dfs
	if isbarpa_m:
		#barpa = pd.merge(barpa_df_wg, barpa_df_env, on=["time","stn_id"])
		barpa = pd.merge(\
				barpa_df_wg[["time","stn_id","wg10_2p2_rad","wg10_2p2_point"]],\
				barpa_df_env,\
			    left_on=["time","stn_id"],right_on=["day","stn_id"])
	else:
		barpa = barpa_df_env
		barpa["wg10_2p2_rad"] = np.nan
		barpa["wg10_2p2_point"] = np.nan
	#merged = pd.merge(dmax, barpa, left_index=True, right_on=["stn_id","time"])
	merged = pd.merge(dmax, barpa, left_on=["dt_floor_6H","stn_id"], right_on=["barpa_env_time","stn_id"])
	merged = pd.merge(merged, barra_df_wg, on=["stn_id","time"])

	return merged

	    
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-start_year', type=str)
	parser.add_argument('-rid', type=str)
	parser.add_argument('-state', type=str)
	args = parser.parse_args()
	start_year = args.start_year
	rid = args.rid
	state = args.state

	end_year = 2015
	date1 = dt.datetime(int(start_year), 1, 1)

	output_df = pd.DataFrame()

	while date1.year <= end_year:
		date2 = last_day_of_month(date1)
		fid1 = date1.strftime("%Y%m%d")
		fid2 = date2.strftime("%Y%m%d")
		fid = rid+"_"+fid1+"_"+fid2
		print(fid1)

		output_df = pd.concat([output_df,load_tint_aws_barpa(fid, state, summary="max")],axis=0)

		date1 = date2 + dt.timedelta(days=1)

	output_df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpa_barra_aws_"+rid+".csv")

	#Notes on df columns
	# -> gust: measured daily max AWS gust speed (in10km if convective, SCW if severe convective gust)
	# -> wg10_barra_12km_rad: BARRA 10 m gust in 50 km radius around AWS
	# -> wg10_barra_12km_point: BARRA 10 m gust at closest point to AWS
	# -> wg10_12km_rad: 6-hourly BARPA-EASTAUS 10 m gust in 50 km radius around AWS
	# -> wg10_12km_point: 6-hourly BARPA-EASTAUS 10 m gust at closest point to AWS
	# -> wg10_12km_rad_dmax: Daily maximum BARPA-EASTAUS 10 m gust in 50 km radius around AWS
	# -> wg10_12km_point_dmax: Daily maximum BARPA-EASTAUS 10 m gust at closest point to AWS
	# -> wg10_2p2_rad: BARPA-M 10 m gust in 10 km radius around AWS
	# -> wg10_2p2_point: BARPA-M 10 m gust at closest point to AWS

	# -> *convective_diagnostic*: A set of convective diagnostics from 12 km BARPA data, at the previous 6-hourly time step before the observed gust
	# -> *convective_diagnostic*_dmax: Daily maximum convective diagnostics from 12 km BARPA data
