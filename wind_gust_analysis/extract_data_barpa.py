from merge_data_barpa import load_barpac_m_wg
from merge_data import last_day_of_month, latlon_dist, load_stn_info, subset_era5, extract_era5_df, get_env_clusters
from merge_data_barpa import get_points, barpac_stations
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
from post_process_tracks import return_drop_list
import GadiClient
import argparse  

if __name__ == "__main__":

	#This script is similar to merge_data_barpa.py. 
	#Except that instead of merging obs and BARPA, we are just extracting BARPAC wind gusts at point locations
	#We are still extracting data at station locations on a state basis though.

	client = GadiClient.GadiClient()

	#Define options and time period (fid)
	start_year = 2005
	end_year = 2015
	date1 = dt.datetime(int(start_year), 1, 1)
	domain = "m"	
	parser = argparse.ArgumentParser()
	parser.add_argument('-state', type=str)
	args = parser.parse_args()
	state = args.state

	forcing = "era"
	#forcing = "cmip5"

	if forcing == "era":
		model = "erai"
		experiment = "historical"
		ensemble = "r0"
	elif forcing == "cmip5":
		model = "ACCESS1-0"
		experiment = "historical"
		ensemble = "r1i1p1"

	#Define station lats and lons
	stn_info = barpac_stations(load_stn_info(state),domain)
	stn_info = stn_info.loc[np.in1d(stn_info.stn_no,return_drop_list(state),invert=True)]
	stn_lat = stn_info["lat"].values
	stn_lon = stn_info["lon"].values
	stn_list = stn_info["stn_no"].values

	#Initialise output df
	output_df = pd.DataFrame()

	#Note that the ERA-forced data starts in December 1990. ACCESS-forced data goes to the end of 2015
	while date1.year <= end_year:
		date2 = last_day_of_month(date1)
		if (date1 >= dt.datetime(1990,12,1)) & (date1.month in [12,1,2]) & (date1 < dt.datetime(2015,3,1)):
				fid1 = date1.strftime("%Y%m%d")
				fid2 = date2.strftime("%Y%m%d")
				fid = "0_"+fid1+"_"+fid2
				print(fid1)
		else:
				date1 = date2 + dt.timedelta(days=1)
				continue

		#Load the gust data for BARPAC-M	
		barpa_wg, x_wg, y_wg = load_barpac_m_wg(fid,forcing,model,experiment,ensemble)
		barpa_wg = barpa_wg.rename({"longitude":"lon","latitude":"lat"})
		barpa_wg = barpa_wg.assign_coords({"time":barpa_wg.time_bnds.values[:,1]})
		barpa_wg_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/lnd_mask-BARPAC-M_km2p2.nc")

		#Subset to within 50 km of stations and then extract the point data
		barpa_subset_wg, rad_lats_wg, rad_lons_wg = subset_era5(barpa_wg, stn_lat, stn_lon, y_wg, x_wg, r=50)
		barpa_wg_lsm = barpa_wg_lsm.interp({"latitude":barpa_subset_wg.lat, "longitude":barpa_subset_wg.lon}, method="nearest").lnd_mask.values
		barpa_subset_wg = xr.where(barpa_wg_lsm, barpa_subset_wg.max_wndgust10m, np.nan).persist()
		barpa_df_wg_point = get_points(barpa_subset_wg, barpa_wg_lsm, stn_lat, stn_lon, stn_list)
		barpa_df_wg_point["time"] = barpa_df_wg_point.time.round("Min")

		#Tidy the dataframe up a bit
		barpa_df_wg_point.drop = barpa_df_wg_point.drop(columns=["forecast_period","forecast_reference_time","height"]).set_index("time")

		#Add to output dataframe		
		output_df = pd.concat([output_df,barpa_df_wg_point],axis=0)

		#Advance to next month
		date1 = date2 + dt.timedelta(days=1)

	#Save point data somewhere
	output_df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_"+model+"_"+experiment+"_"+state+"_"+str(start_year)+"_"+str(end_year)+".csv")


