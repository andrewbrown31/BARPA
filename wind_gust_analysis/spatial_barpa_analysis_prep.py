import xarray as xr
import GadiClient 
import dask
import glob
import numpy as np
import dask
from merge_data import last_day_of_month
import os
import datetime as dt
import pandas as pd
from merge_data import get_env_clusters, last_day_of_month
from era5_spatial_cluster import transform_era5, era5_clustering


def load_hist(y1, y2):
    #Load the ACCESS-forced historical BARPA data. BARPA-R data is from wrf_non_parallel.py. BARPAC data is from barpa_spatial_scw.py
    files_barpar = []
    files_barpac = []
    for y in np.arange(y1,y2+1):
        for m in [12,1,2]:
            date = dt.datetime(y,m,1)
            path_barpar = "/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpa_access_"+date.strftime("%Y%m%d")+"_"+last_day_of_month(dt.datetime(y,m,1)).strftime("%Y%m%d")+".nc"
            path_barpac = "/scratch/eg3/ab4502/barpa_scw_ACCESS1-0_historical_0_"+date.strftime("%Y%m%d")+"_"+last_day_of_month(dt.datetime(y,m,1)).strftime("%Y%m%d")+".nc"        
            if (date>=dt.datetime(1985,12,1)) & (date < dt.datetime(2005,3,1)):
                if (os.path.isfile(path_barpar)):
                    files_barpar.append(path_barpar)
                if (os.path.isfile(path_barpac)):                
                    files_barpac.append(path_barpac)            
            else:
                print("Skipping "+date.strftime("%Y%m%d")+"...")

    #Load BARPAR
    v = ["qmean01","lr13","Umean06","s06","bdsd"]
    barpar_hist = xr.open_mfdataset(files_barpar)[v].sel({"lat":slice(-44.47,-28.9),"lon":slice(135.5,156.0)})
    
    #Do environmental clustering
    cluster_mod, cluster_input = get_env_clusters()
    s06, qmean01, lr13, Umean06 = transform_era5(barpar_hist, cluster_mod, cluster_input)
    s06 = s06.persist(); qmean01 = qmean01.persist(); lr13 = lr13.persist(); Umean06 = Umean06.persist()
    cluster = era5_clustering(s06, qmean01, lr13, Umean06, barpar_hist, cluster_mod).persist()

    #Mask ocean
    lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").rename({"latitude":"lat","longitude":"lon"}).interp_like(cluster,method="nearest")
    cluster = xr.where(lsm.lnd_mask==1,cluster,np.nan)
    
    #Load barpac
    barpac_hist = xr.open_mfdataset(files_barpac).sel({"latitude":slice(-44.47,-28.9),"longitude":slice(135.5,156.0)})
    
    return output_bdsd(cluster), barpac_hist

def load_rcp(y1, y2):
    files_barpar = []
    files_barpac = []
    for y in np.arange(y1,y2+1):
        for m in [12,1,2]:
            date = dt.datetime(y,m,1)
            path_barpar = "/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpa_access_"+date.strftime("%Y%m%d")+"_"+last_day_of_month(dt.datetime(y,m,1)).strftime("%Y%m%d")+".nc"
            path_barpac = "/scratch/eg3/ab4502/barpa_scw_ACCESS1-0_rcp85_0_"+date.strftime("%Y%m%d")+"_"+last_day_of_month(dt.datetime(y,m,1)).strftime("%Y%m%d")+".nc"        
            if (date>=dt.datetime(2039,12,1)) & (date < dt.datetime(2059,3,1)):
                if (os.path.isfile(path_barpar)):
                    files_barpar.append(path_barpar)
                if (os.path.isfile(path_barpac)):                
                    files_barpac.append(path_barpac)            
            else:
                print("Skipping "+date.strftime("%Y%m%d")+"...")                

    #Load BARPAR
    v = ["qmean01","lr13","Umean06","s06","bdsd"]
    barpar_hist = xr.open_mfdataset(files_barpar)[v].sel({"lat":slice(-44.47,-28.9),"lon":slice(135.5,156.0)})
    
    #Do environmental clustering
    cluster_mod, cluster_input = get_env_clusters()
    s06, qmean01, lr13, Umean06 = transform_era5(barpar_hist, cluster_mod, cluster_input)
    s06 = s06.persist(); qmean01 = qmean01.persist(); lr13 = lr13.persist(); Umean06 = Umean06.persist()
    cluster = era5_clustering(s06, qmean01, lr13, Umean06, barpar_hist, cluster_mod).persist()

    #Mask ocean
    lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").rename({"latitude":"lat","longitude":"lon"}).interp_like(cluster,method="nearest")
    cluster = xr.where(lsm.lnd_mask==1,cluster,np.nan)
    
    #Load barpac
    barpac_hist = xr.open_mfdataset(files_barpac).sel({"latitude":slice(-44.47,-28.9),"longitude":slice(135.5,156.0)})
    
    return output_bdsd(cluster), barpac_hist

def scw_mask(ds, wg_thresh, wgr_thresh, cluster, ds_lsm):
    
    #Get SCW events from BARPAC based on a wind gust and wind gust ratio threshold
    
    if cluster=="all":
        scws = xr.where(((ds.gust_dmax >= wg_thresh) & (ds.wgr_dmax >= wgr_thresh)),1,0)
    else:
        scws = xr.where(((ds.gust_dmax >= wg_thresh) & (ds.wgr_dmax >= wgr_thresh) & (ds.cluster==cluster)),1,0)
    scws = xr.where(ds_lsm.lnd_mask==1, scws, np.nan).persist()
                        
    return scws

def get_scws(ds, wg_thresh, wgr_thresh, interp_lon, interp_lat):
    
    #Get SCW events (binary mask), for each cluster
    barpac_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/lnd_mask-BARPAC-M_km2p2.nc").sel({"latitude":slice(-44.47,-28.9),"longitude":slice(135.5,156.1)})    
    scws_all = scw_mask(ds, wg_thresh, wgr_thresh, "all", barpac_lsm)
    scws0 = scw_mask(ds, wg_thresh, wgr_thresh, 0, barpac_lsm)
    scws1 = scw_mask(ds, wg_thresh, wgr_thresh, 1, barpac_lsm)                        
    scws2 = scw_mask(ds, wg_thresh, wgr_thresh, 2, barpac_lsm)                            
    scws = xr.Dataset({"cluster0_events":scws0, "cluster1_events":scws1, "cluster2_events":scws2, "all_events":scws_all})
                        
    #Interpolare BARPAC SCW events to the BARPA-R grid
    dx = dy = 0.11
    lon_bins = np.insert(interp_lon - (0.11 / 2), interp_lon.shape[0], [interp_lon[-1] + 0.11 / 2])
    lat_bins = np.insert(interp_lat - (0.11 / 2), interp_lat.shape[0], [interp_lat[-1] + 0.11 / 2])          
    scws = scws.groupby_bins("longitude",bins=lon_bins,include_lowest=True).max().groupby_bins("latitude",bins=lat_bins,include_lowest=True).max()
    scws = scws.assign_coords({"latitude_bins":interp_lat, "longitude_bins":interp_lon})
    scws = scws.rename({"latitude_bins":"lat","longitude_bins":"lon"}).persist()

    #mask ocean
    lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").rename({"latitude":"lat","longitude":"lon"}).interp_like(scws,method="nearest")
    
    return xr.where(lsm.lnd_mask==1,scws,np.nan)

def get_max_gust(ds, wg_thresh, wgr_thresh, interp_lon, interp_lat):
    
    #Get the maximum SCW gust on the BARPA-R grid, for each cluster.
    barpac_lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPAC-M_km2p2/static/lnd_mask-BARPAC-M_km2p2.nc").sel({"latitude":slice(-44.47,-28.9),"longitude":slice(135.5,156.1)})    
    max_gust_all = xr.where(((ds.gust_dmax >= wg_thresh) & (ds.wgr_dmax >= wgr_thresh)),ds.gust_dmax,0)
    max_gust0 = xr.where(((ds.gust_dmax >= wg_thresh) & (ds.wgr_dmax >= wgr_thresh) & (ds.cluster==0)),ds.gust_dmax,np.nan)    
    max_gust1 = xr.where(((ds.gust_dmax >= wg_thresh) & (ds.wgr_dmax >= wgr_thresh) & (ds.cluster==1)),ds.gust_dmax,np.nan)    
    max_gust2 = xr.where(((ds.gust_dmax >= wg_thresh) & (ds.wgr_dmax >= wgr_thresh) & (ds.cluster==2)),ds.gust_dmax,np.nan)    
    max_gust = xr.Dataset({"cluster0_max":max_gust0, "cluster1_max":max_gust1, "cluster2_max":max_gust2, "max":max_gust_all})
    max_gust = xr.where(barpac_lsm.lnd_mask==1, max_gust, np.nan).persist()
    
    #Interpolare BARPAC SCW events to the BARPA-R grid
    dx = dy = 0.11
    lon_bins = np.insert(interp_lon - (0.11 / 2), interp_lon.shape[0], [interp_lon[-1] + 0.11 / 2])
    lat_bins = np.insert(interp_lat - (0.11 / 2), interp_lat.shape[0], [interp_lat[-1] + 0.11 / 2])          
    max_gust = max_gust.groupby_bins("longitude",bins=lon_bins,include_lowest=True).max().groupby_bins("latitude",bins=lat_bins,include_lowest=True).max()
    max_gust = max_gust.assign_coords({"latitude_bins":interp_lat, "longitude_bins":interp_lon})
    max_gust = max_gust.rename({"latitude_bins":"lat","longitude_bins":"lon"}).persist()

    #mask ocean
    lsm = xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").rename({"latitude":"lat","longitude":"lon"}).interp_like(max_gust,method="nearest")
    
    return xr.where(lsm.lnd_mask==1,max_gust,np.nan)
    
def output_bdsd(ds):
    
    dim=("lat","lon","date")
    out_ds = xr.Dataset({
            "cluster0":(dim, (ds.cluster==0).groupby(ds["time"].dt.date).max().values),
            "cluster1":(dim, (ds.cluster==1).groupby(ds["time"].dt.date).max().values),
            "cluster2":(dim, (ds.cluster==2).groupby(ds["time"].dt.date).max().values),    
            "clusterall_bdsd":(dim, (ds.bdsd>=0.83).groupby(ds["time"].dt.date).max().values),    
            "cluster0_bdsd":(dim, ((ds.cluster==0) & (ds.bdsd>=0.83)).groupby(ds["time"].dt.date).max().values),
            "cluster1_bdsd":(dim, ((ds.cluster==1) & (ds.bdsd>=0.83)).groupby(ds["time"].dt.date).max().values),
            "cluster2_bdsd":(dim, ((ds.cluster==2) & (ds.bdsd>=0.83)).groupby(ds["time"].dt.date).max().values),
        },
        coords={"lat":(("lat"), ds.lat.values), "lon":(("lon"), ds.lon.values), "date":np.unique(ds.time.dt.date.values)}) 
    
    return out_ds

def monthly_resample(ds,func):
    
    year_month_idx = [str(a) + str(b).zfill(2) for a, b in zip (pd.DatetimeIndex(ds.date.values).year, pd.DatetimeIndex(ds.date.values).month)]
    ds.coords['year_month'] = ('date', year_month_idx)
    time = list(pd.DatetimeIndex(ds.date.values).strftime("%Y%m%d"))
    if func=="sum":
        ds_monthly = ds.groupby('year_month').sum(skipna=False)
    elif func=="max":        
        ds_monthly = ds.groupby('year_month').max()        
    ds_monthly.attrs["daily_time"] = time 
    return ds_monthly

if __name__ == "__main__":

	#Set up dask
	GadiClient.GadiClient()
	dask.config.set({'array.slicing.split_large_chunks': False})

	#Load BARPAC and BARPA-R environmental data for both periods
	print("LOADING DATA...")
	barpar_hist, barpac_hist = load_hist(1985,2005)
	barpar_rcp, barpac_rcp = load_rcp(2039,2059)

	#Define SCW events from BARPAC and re-grid to the BARPA-R grid
	wg_thresh=25
	wgr_thresh=1.5
	interp_lon=barpar_hist.lon.values
	interp_lat=barpar_hist.lat.values
	print("DEFINING SCW EVENTS AND MAX GUSTS...")
	scws_hist = get_scws(barpac_hist, wg_thresh, wgr_thresh, interp_lon, interp_lat)
	scws_rcp = get_scws(barpac_rcp, wg_thresh, wgr_thresh, interp_lon, interp_lat)
	max_hist = get_max_gust(barpac_hist, wg_thresh, wgr_thresh, interp_lon, interp_lat)
	max_rcp = get_max_gust(barpac_rcp, wg_thresh, wgr_thresh, interp_lon, interp_lat)

	#Now we have daily, on BARPA-R grid, for each cluster:
	# -> BDSD occurrences from BARPA-R
	# -> SCW occurrences
	# -> Max gust
	#Resample to monthly, and save
	print("RESAMPLE TO MONTHLY AND SAVE...")
	scws_hist_monthly = monthly_resample(scws_hist, "sum")
	scws_rcp_monthly = monthly_resample(scws_rcp, "sum")
	max_hist_monthly = monthly_resample(max_hist, "max")
	max_rcp_monthly = monthly_resample(max_rcp, "max")
	barpar_hist_monthly = monthly_resample(barpar_hist, "sum")
	barpar_rcp_monthly = monthly_resample(barpar_rcp, "sum")
	
	#TODO: SAVE THE OUTPUT TO GDATA, RUN AS A QSUB
	scws_hist_monthly.to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpac_scws_hist_monthly.nc")
	scws_rcp_monthly.to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpac_scws_rcp_monthly.nc")
	max_hist_monthly.to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpac_max_hist_monthly.nc")
	max_rcp_monthly.to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpac_max_rcp_monthly.nc")
	barpar_hist_monthly.to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpar_hist_monthly.nc")
	barpar_rcp_monthly.to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/barpa_access/barpar_rcp_monthly.nc")
	
