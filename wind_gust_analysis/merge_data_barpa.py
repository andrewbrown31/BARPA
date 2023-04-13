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
from post_process_tracks import load_aws, return_drop_list
from GadiClient import GadiClient

def load_barra_env_wg10(fid):
	barra_str = fid.split("_")[1]+"_"+fid.split("_")[2]
	barra = xr.open_mfdataset("/g/data/cj37/BARRA/BARRA_R/v1/forecast/spec/max_wndgust10m/"+\
                fid.split("_")[1][0:4] + "/" + fid.split("_")[1][4:6] + "/max_wndgust10m*" + fid.split("_")[1][0:6] + "*.nc",\
		    combine="nested",concat_dim="time").sortby("time")
	barra_lat = barra["latitude"].values
	barra_lon = barra["longitude"].values
	x,y = np.meshgrid(barra_lon,barra_lat)
	return barra, x, y

def load_barpa_r_wg(fid):

	barpa_str1 = fid.split("_")[1]
	barpa_str2 = (dt.datetime.strptime(fid.split("_")[1],"%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")

	files1 = glob.glob("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/era/erai/historical/r0/*/"+barpa_str1[0:6]+"*/pp26/wndgust10m-pp26-BARPA-EASTAUS_12km*.nc")
	files2 = glob.glob("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/era/erai/historical/r0/*/"+barpa_str2[0:6]+"*/pp26/wndgust10m-pp26-BARPA-EASTAUS_12km*.nc")

	barpa = xr.open_mfdataset(files2+files1,combine="nested",concat_dim="time").drop_duplicates("time").sortby("time").\
		sel({"time":slice(dt.datetime.strptime(fid.split("_")[1],"%Y%m%d").replace(minute=10),\
				  dt.datetime.strptime(fid.split("_")[2],"%Y%m%d") + dt.timedelta(days=1))})

	barpa_lat = barpa["latitude"].values
	barpa_lon = barpa["longitude"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, x, y

def load_barpa_env(fid):
	barpa_str = fid.split("_")[1]+"_"+fid.split("_")[2]
	barpa = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_erai/barpa_erai_"+barpa_str+".nc", chunks={"time":16})
	barpa_lat = barpa["lat"].values
	barpa_lon = barpa["lon"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, x, y

def load_barpac_m_lightning(fid):
    
	#Load 10-minute BARPAC-M data from /scratch. Also resample to hourly maximum.
    
	barpa_str1 = fid.split("_")[1]
	barpa_str2 = (dt.datetime.strptime(fid.split("_")[1],"%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")

	files1 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str1[0:6]+"*/pp0/n_lightning_fl-pp0-BARPAC-M_km2p2-*.nc")
	files2 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str2[0:6]+"*/pp0/n_lightning_fl-pp0-BARPAC-M_km2p2-*.nc")

	barpa = xr.open_mfdataset(files2+files1,combine="nested",concat_dim="time").drop_duplicates("time").sortby("time")\
		    .sel({"time":slice(dt.datetime.strptime(fid.split("_")[1],"%Y%m%d"),\
				       dt.datetime.strptime(fid.split("_")[2],"%Y%m%d") + dt.timedelta(days=1))})

	barpa_lat = barpa["latitude"].values
	barpa_lon = barpa["longitude"].values
	x,y = np.meshgrid(barpa_lon,barpa_lat)
	return barpa, x, y

def load_barpac_m_wg(fid):
    
	#Load 10-minute BARPAC-M data from /scratch. Also resample to hourly maximum.
    
	barpa_str1 = fid.split("_")[1]
	barpa_str2 = (dt.datetime.strptime(fid.split("_")[1],"%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")

	files1 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str1[0:6]+"*/pp26/max_wndgust10m-pp26-BARPAC-M_km2p2-*.nc")
	files2 = glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+barpa_str2[0:6]+"*/pp26/max_wndgust10m-pp26-BARPAC-M_km2p2-*.nc")

	barpa = xr.open_mfdataset(files2+files1)\
		    .sel({"time":slice(dt.datetime.strptime(fid.split("_")[1],"%Y%m%d"),\
				       dt.datetime.strptime(fid.split("_")[2],"%Y%m%d") + dt.timedelta(days=1))})

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

def add_rolling_mean(df,time_col="time",gust_col="wndgust10m",min_periods=6):
    rolling=df.set_index(time_col).groupby("stn_id")[gust_col].rolling("4H",center=True,closed="both",min_periods=min_periods).mean()
    df=pd.merge(df,rolling.rename("rolling_4hr_mean"),on=["stn_id",time_col])
    df["wgr_4"] = df[gust_col] / df["rolling_4hr_mean"]
    return df

def extract_lightning_points(lightning, stn_lat, stn_lon, stn_list):

	lon = lightning.lon.values
	lat = lightning.lat.values
	x, y = np.meshgrid(lon,lat)
	df_out = pd.DataFrame()
	for i in np.arange(len(stn_lat)):
		dist_km = latlon_dist(stn_lat[i], stn_lon[i], y, x)
		sliced = xr.where(dist_km<=50, lightning, np.nan)
		temp = sliced.sum(("lat","lon")).Lightning_observed.to_dataframe()
		temp["points"] = i
		df_out = pd.concat([df_out,temp], axis=0)

	for p in np.arange(len(stn_list)):
		df_out.loc[df_out.points==p,"stn_id"] = stn_list[p]
	df_out = df_out.drop("points",axis=1)

	return df_out

def load_lightning(fid):
	yyyymmdd1 = fid.split("_")[1]
	yyyymmdd2 = fid.split("_")[2]
	start_t = dt.datetime(int(yyyymmdd1[0:4]), int(yyyymmdd1[4:6]), int(yyyymmdd1[6:8]))
	end_t = dt.datetime(int(yyyymmdd2[0:4]), int(yyyymmdd2[4:6]), int(yyyymmdd2[6:8]))

	f = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_1hr/lightning_AUS_0.25deg_1hr_"+yyyymmdd1[0:4]+".nc",\
	    decode_times=False)
	f = f.assign_coords(time=[dt.datetime(int(yyyymmdd1[0:4]),1,1,0) + dt.timedelta(hours=int(i)) for i in np.arange(len(f.time.values))]).\
                sel({"time":slice(start_t,end_t.replace(hour=23))})
	return f

def barpac_stations(stn_info,domain):
    
    if domain=="m":
        f = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/topog-BARPAC-M_km2p2.nc")
    elif domain=="t":
        f = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-T_km4p4/static/topog-BARPAC-T_km4p4.nc")
    else:
        raise ValueError("Domain must be m or t")
        
    
    lon_bnds = [f.longitude.values.min(),f.longitude.values.max()]
    lat_bnds = [f.latitude.values.min(),f.latitude.values.max()]    
    
    return stn_info.loc[(stn_info.lon >= lon_bnds[0]) & (stn_info.lon <= lon_bnds[1]) & (stn_info.lat >= lat_bnds[0]) & (stn_info.lat <= lat_bnds[1])]


def load_tint_aws_barpa(fid, state):

	#########################################################################
	#   Observations							#
	#########################################################################

	#Instead of loading just the TINT-processed AWS data, load all the AWS data for a particular state.
	domain="m"
	year=fid.split("_")[1][0:4]
	stn_info = barpac_stations(load_stn_info(state),domain)
	stn_info = stn_info.loc[np.in1d(stn_info.stn_no,return_drop_list(state),invert=True)]
	aws = load_aws(state, year)
	aws = aws[aws["q"]=="Y"]
	aws = aws[np.in1d(aws.stn_id,stn_info.stn_no)].drop(columns=["dt_utc","q","dt_lt"])

	#Resample to 10-minute maximum to align with BARPA
	aws = aws.groupby("stn_id")[["gust"]].resample("10Min",closed="right",label="right").max()

	#Calculate the 4-hour wind gust ratio, for our SCW definition
	rolling4 = \
	    aws.reset_index(0).groupby("stn_id").gust.rolling("4H",center=True,closed="both",min_periods=6).mean()
	rolling4.name = "rolling4"
	aws = pd.concat([aws,rolling4],axis=1)
	aws["wgr_4"] = aws["gust"] / aws["rolling4"]

	#Resample to daily max
	aws["dt_floor_1D"] = aws.index.get_level_values(1).floor("1D")

	#Load AWS station information
	stn_list = np.unique(aws.index.get_level_values(0))
	stn_latlon = stn_info.set_index("stn_no").loc[stn_list][["lat","lon"]].values
	stn_lat = stn_latlon[:,0]; stn_lon = stn_latlon[:,1]


	#########################################################################
	#   BARPA-R convective diagnostics (12 km)				#
	#########################################################################

	#Load BARPA 12 km environmental diagnostics
	barpa_env, x_env, y_env = load_barpa_env(fid)

	#Subset to within 50 km
	barpa_subset_env, rad_lats_env, rad_lons_env = subset_era5(barpa_env, stn_lat, stn_lon, y_env, x_env)

	#Mask ocean
	barpa_env_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").\
			    interp({"latitude":barpa_subset_env.lat, "longitude":barpa_subset_env.lon}, method="nearest").lnd_mask.values
	barpa_subset_env = xr.where(barpa_env_lsm, barpa_subset_env, np.nan).persist()

	#Get dataframe from the BARPA data
	barpa_df_env = extract_era5_df(barpa_subset_env, rad_lats_env, rad_lons_env, stn_list,"max")

	#Load clustering classification model saved by ~/working/observations/tint_processing/auto_case_driver/kmeans_and_cluster_eval.ipynb
	#Order = [1,2,0] for clusters = [strong bg wind, steep lapse rate, high moisture]
	cluster_mod, cluster_input = get_env_clusters()
	input_df = (barpa_df_env[["s06","qmean01","lr13","Umean06"]]\
		   - cluster_input.min(axis=0))\
	    / (cluster_input.max(axis=0) - \
	       cluster_input.min(axis=0))
	barpa_df_env["cluster"] = cluster_mod.predict(input_df)

	#########################################################################
	#   BARPA-R wind gust (12 km)						#
	#########################################################################

	#Load 12 km wind gusts
	barpa_r_wg, barpa_r_wg_x, barpa_r_wg_y = load_barpa_r_wg(fid)
	barpa_r_wg = barpa_r_wg.rename({"longitude":"lon","latitude":"lat"})

	#Subset to within 50 km
	barpa_subset_r_wg, rad_lats_r_wg, rad_lons_r_wg = subset_era5(barpa_r_wg, stn_lat, stn_lon, barpa_r_wg_y, barpa_r_wg_x, r=50)

	#Mask ocean
	barpa_r_wg_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").\
                            interp({"latitude":barpa_subset_r_wg.lat, "longitude":barpa_subset_r_wg.lon}, method="nearest").lnd_mask.values
	barpa_subset_r_wg = xr.where(barpa_r_wg_lsm, barpa_subset_r_wg.wndgust10m, np.nan).persist()

	#Get dataframe from the BARPA data (50 km radius)
	barpa_df_r_wg = extract_era5_df(barpa_subset_r_wg, rad_lats_r_wg, rad_lons_r_wg, stn_list,"max")

	#Get dataframe for the closet land point
	barpa_df_r_wg_point = get_points(barpa_subset_r_wg, barpa_r_wg_lsm, stn_lat, stn_lon, stn_list)

	#Round to nearest 10 min
	barpa_df_r_wg["time"] = barpa_df_r_wg.time.round("Min")
	barpa_df_r_wg_point["time"] = barpa_df_r_wg_point.time.round("Min")

	#########################################################################
	#   BARPA (2.2 km)							#
	#########################################################################

	#Put in a check for BARPA data
	isbarpa_m = len(glob.glob("/scratch/tp28/cst565/ESCI/BARPAC-M_km2p2/era/erai/historical/r0/*/"+\
                fid.split("_")[1][0:4]+"*")) > 0
	if isbarpa_m:

		#Load the BARPA 2.2 km wind gusts
		barpa_wg, x_wg, y_wg = load_barpac_m_wg(fid)
		barpa_wg = barpa_wg.rename({"longitude":"lon","latitude":"lat"})
		barpa_wg = barpa_wg.assign_coords({"time":barpa_wg.time_bnds.values[:,1]})

		#Subset to within r km
		barpa_subset_wg, rad_lats_wg, rad_lons_wg = subset_era5(barpa_wg, stn_lat, stn_lon, y_wg, x_wg, r=50)

		#Mask ocean
		barpa_wg_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/lnd_mask-BARPAC-M_km2p2.nc").\
				    interp({"latitude":barpa_subset_wg.lat, "longitude":barpa_subset_wg.lon}, method="nearest").lnd_mask.values
		barpa_subset_wg = xr.where(barpa_wg_lsm, barpa_subset_wg.max_wndgust10m, np.nan).persist()

		#Get dataframe from the BARPA data
		barpa_df_wg = extract_era5_df(barpa_subset_wg, rad_lats_wg, rad_lons_wg, stn_list,"max")
		#barpa_df_wg["time"] = barpa_df_wg.time.dt.floor("1D")

		#Get dataframe using the closest point function. Merge them with the other dataframes
		barpa_df_wg_point = get_points(barpa_subset_wg, barpa_wg_lsm, stn_lat, stn_lon, stn_list)

		#Round to nearest 10 min
		barpa_df_wg["time"] = barpa_df_wg.time.round("Min")
		barpa_df_wg_point["time"] = barpa_df_wg_point.time.round("Min")

	#########################################################################
	#   Observed lightning							#
	#########################################################################
	if (int(year) >= 2005) & (int(year) <= 2020):
		obs_lightning = load_lightning(fid)
		obs_lightning_df = extract_lightning_points(obs_lightning, stn_lat, stn_lon, stn_list).reset_index()
		dmax_obs_lightning = obs_lightning_df.groupby("stn_id").resample("1D",on="time")[["Lightning_observed"]].sum()
	else:
		print("NOTE THAT LIGHTNING DATA IS NOT AVAILABLE FOR THIS YEAR")

	#########################################################################
	#   BARPA lightning							#
	#########################################################################
	if isbarpa_m:
		barpa_lightning,x_barpa_lightning,y_barpa_lightning = load_barpac_m_lightning(fid)
		barpa_lightning = barpa_lightning.rename({"longitude":"lon","latitude":"lat"}).drop_vars(["time_bnds"]).drop_dims("bnds")

		#Subset to within r km
		barpa_subset_lightning, rad_lats_lightning, rad_lons_lightning = \
			subset_era5(barpa_lightning, stn_lat, stn_lon, y_barpa_lightning, x_barpa_lightning, r=50)

		#Get dataframe from the BARPA data
		barpa_df_lightning = extract_era5_df(barpa_subset_lightning, rad_lats_lightning, rad_lons_lightning, stn_list,"sum")
		barpa_df_lightning["time"] = barpa_df_lightning.time.dt.floor("1D")

	#########################################################################
	#   Merge together							#
	#########################################################################

	# OBS: hmax
	# BARPA-R: barpa_df_env, barpa_df_r_wg, barpa_df_r_wg_point
	# BARPAC-M: barpa_df_wg, barpa_df_wg_point
	# BARRA-R: barra_df_wg, barra_df_wg10_point

	#Merge the hourly max obs with the 6-hourly barpa environmental data
	aws["dt_floor_6H"] = aws.index.get_level_values(1).floor("6H")
	merged = pd.merge(aws.reset_index(), barpa_df_env, left_on=["stn_id","dt_floor_6H"], right_on=["stn_id","time"], how="inner").set_index(["stn_id","dt_utc"])

	#Resample and calculate WGR: BARPA-R
	barpa_df_r_wg_resample = add_rolling_mean(barpa_df_r_wg,gust_col="wndgust10m").\
				    drop(columns=["forecast_period","forecast_reference_time","height","rolling_4hr_mean"]).\
				    rename(columns={"wndgust10m":"wg10_12km_rad","wgr_4":"wgr_12km_rad"})
	barpa_df_r_wg_point_resample = add_rolling_mean(barpa_df_r_wg_point,gust_col="wndgust10m").\
				    drop(columns=["lat","lon","forecast_period","forecast_reference_time","height","rolling_4hr_mean"]).\
				    rename(columns={"wndgust10m":"wg10_12km_point","wgr_4":"wgr_12km_point"})

	#Resample and calculate WGR: BARPAC-M
	barpa_df_wg_resample = add_rolling_mean(barpa_df_wg,gust_col="max_wndgust10m").\
				    drop(columns=["forecast_period","forecast_reference_time","height","rolling_4hr_mean"]).\
				    rename(columns={"max_wndgust10m":"wg10_2p2km_rad","wgr_4":"wgr_2p2km_rad"})
	barpa_df_wg_point_resample = add_rolling_mean(barpa_df_wg_point,gust_col="max_wndgust10m").\
				    drop(columns=["forecast_period","forecast_reference_time","height","rolling_4hr_mean","lat","lon"]).\
				    rename(columns={"max_wndgust10m":"wg10_2p2km_point","wgr_4":"wgr_2p2km_point"})

	merged = pd.concat([merged,\
			    barpa_df_r_wg_resample.set_index(["stn_id","time"]),\
			    barpa_df_r_wg_point_resample.set_index(["stn_id","time"]),\
			    barpa_df_wg_resample.set_index(["stn_id","time"]),\
			    barpa_df_wg_point_resample.set_index(["stn_id","time"])],axis=1,join="inner")

	#Add daily max BARPA/observed lightning
	dmax_lightning_merged = pd.concat([barpa_df_lightning.set_index(["stn_id","time"]).drop(columns=["latitude_longitude","forecast_period","forecast_reference_time"]),
					    dmax_obs_lightning],axis=1,join="inner")
	merged = pd.merge(merged,dmax_lightning_merged.rename(columns={"Lightning_observed":"Lightning_observed_daily"}),
		    left_on=["stn_id","dt_floor_1D"],right_index=True,how="inner")


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
		if (date1.month in [12,1,2]) & (date1 < dt.datetime(2015,3,1)):
			fid1 = date1.strftime("%Y%m%d")
			fid2 = date2.strftime("%Y%m%d")
			fid = rid+"_"+fid1+"_"+fid2
			print(fid1)

			output_df = pd.concat([output_df,load_tint_aws_barpa(fid, state)],axis=0)

		date1 = date2 + dt.timedelta(days=1)

	output_df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_"+state+".csv")

	#Notes on df columns
	# -> gust: measured 10-min max AWS gust speed
	# -> wg10_12km_rad: 10-min BARPA-EASTAUS 10 m gust in 50 km radius around AWS
	# -> wg10_12km_point: 10-min max BARPA-EASTAUS 10 m gust at closest point to AWS
	# -> wg10_2p2_rad: 10-min max BARPAC gust in 10 km radius around AWS
	# -> wg10_2p2_point: 10-min max BARPAC 10 m gust at closest point to AWS
	# -> wgr_*: Ratio of 10-min max gust to the rolling 4-hour mean 

	# -> n_lightning_fl: Sum of BARPAC lightning flashes in all grid points within 50 km of AWS on the same day as the gust
	# -> Lightning_observed_daily: Sum of observed WWLLN lightning flashes in all grid points within 50 km of AWS on the same day as the gust

	# -> *convective_diagnostic*: A set of convective diagnostics from 12 km BARPA data, at the previous 6-hourly time step before the observed gust
