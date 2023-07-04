import numpy as np
import pandas as pd
import datetime as dt
from merge_data_barpa import load_stn_info
import xarray as xr
import tqdm
import warnings

from sharppy.sharptab import winds, utils, params, thermo, interp, profile
from metpy.calc import dewpoint_from_relative_humidity, dewpoint_from_specific_humidity
import glob

def drop_duplicates(ds):

        #Drop time duplicates

        a, ind = np.unique(ds.time.values, return_index=True)
        return(ds.isel({"time":ind}))


def get_var_files(str_fmt,barpa_str1,barpa_str2,var,stream,time):
    
    files = glob.glob(str_fmt % (barpa_str1[0:6],stream,var)) + glob.glob(str_fmt % (barpa_str2[0:6],stream,var))
    file_times = np.array([dt.datetime.strptime(f.split("/")[-1].split("-")[-1].split(".")[0],"%Y%m%dT%H%MZ") for f in files])

    time_diff = np.array([time-t for t in file_times])
    time_diff[[t.total_seconds() < 0 for t in time_diff]] = dt.timedelta(days=10000)

    return files[np.argmin(time_diff)]    
    
def plot_sounding_barpa(time, lat, lon):

    barpa_str1 = time.strftime("%Y%m%d")
    barpa_str2 = (time - dt.timedelta(days=5)).strftime("%Y%m%d")    
    
    str_fmt = "/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/era/erai/historical/r0/*/%s*/%s/%s*.nc"

    zs=xr.open_dataset("/g/data/tp28/BARPA/trials/BARPA-EASTAUS_12km/static/topog-BARPA-EASTAUS_12km.nc").\
                        interp({"longitude":lon, "latitude":lat},method="nearest")["topog"]

    z = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"geop_ht_uv","pp2",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["geop_ht_uv"]
    t = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"air_temp","pp2",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["air_temp_uv"]
    s = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"spec_hum","pp2",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["spec_hum_uv"]
    u = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"wnd_ucmp","pp2",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["wnd_ucmp_uv"]
    v = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"wnd_vcmp","pp2",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["wnd_vcmp_uv"]  
    
    ts = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"temp_scrn","pp3",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["temp_scrn"]
    dps = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"dewpt_scrn","pp26",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["dewpt_scrn"]
    us = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"uwnd10m_b","pp3",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["uwnd10m_b"]
    vs = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"uwnd10m_b","pp3",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["uwnd10m_b"]
    ps = xr.open_dataset(get_var_files(str_fmt,barpa_str1,barpa_str2,"sfc_pres","pp26",time)).sel({"time":time},method="nearest").interp({"longitude":lon, "latitude":lat},method="nearest")["sfc_pres"]


    p=z.pressure
    dp=dewpoint_from_specific_humidity(p,t,s)
    prof_pres = np.flip(p.values.squeeze())
    prof_hgt = np.flip(z.values.squeeze())
    prof_tmpc = np.flip(t.values.squeeze()-273.15)
    prof_dwpc = np.flip(np.array(dp).squeeze())
    prof_u = np.flip(u.values.squeeze()*1.94)
    prof_v = np.flip(v.values.squeeze()*1.94)
    
    #If the surface data plays nicely with the pressure level data, then use it. 
    #Sometimes, the first pressure level has z=0, even though the pressure of that level is less than sfc pressure (should be above ground level)
    agl_inds = prof_pres <= (ps.values/100.)
    if prof_hgt[agl_inds][0] > zs.values:
        prof = profile.create_profile(pres=np.insert(prof_pres[agl_inds], 0, ps.values/100.),
                                      hght=np.insert(prof_hgt[agl_inds], 0, zs.values),
                                      tmpc=np.insert(prof_tmpc[agl_inds], 0, ts.values-273.15),
                                      dwpc=np.insert(prof_dwpc[agl_inds], 0, dps.values-273.15),
                                      u=np.insert(prof_u[agl_inds], 0, us.values*1.94),
                                      v=np.insert(prof_v[agl_inds], 0, vs.values*1.94))
    else:
        print("Unable to use surface data")
        agl_inds = prof_hgt > 0
        prof = profile.create_profile(pres=prof_pres[agl_inds],
                                      hght=prof_hgt[agl_inds],
                                      tmpc=prof_tmpc[agl_inds],
                                      dwpc=prof_dwpc[agl_inds],
                                      u=prof_u[agl_inds],
                                      v=prof_v[agl_inds])    
    
    sfcpcl = params.parcelx( prof, flag=1 )
    mupcl = params.parcelx( prof, flag=3 )
    mlpcl = params.parcelx( prof, flag=4 )   
    return prof, mupcl

def plot_sounding_era5(time, lat, lon):

    str_fmt = "/g/data/rt52/era5/pressure-levels/reanalysis/%s/%d/%s_era5_oper_pl_%s*.nc" 
    str_fmt_sfc = "/g/data/rt52/era5/single-levels/reanalysis/%s/%d/%s_era5_oper_sfc_%s*.nc"
    
    z=xr.open_dataset(glob.glob(str_fmt % ("z",time.year,"z",time.strftime("%Y%m01")))[0])["z"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")
    t=xr.open_dataset(glob.glob(str_fmt % ("t",time.year,"t",time.strftime("%Y%m01")))[0])["t"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")
    r=xr.open_dataset(glob.glob(str_fmt % ("r",time.year,"r",time.strftime("%Y%m01")))[0])["r"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")
    u=xr.open_dataset(glob.glob(str_fmt % ("u",time.year,"u",time.strftime("%Y%m01")))[0])["u"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")
    v=xr.open_dataset(glob.glob(str_fmt % ("v",time.year,"v",time.strftime("%Y%m01")))[0])["v"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")

    ts=xr.open_dataset(glob.glob(str_fmt_sfc % ("2t",time.year,"2t",time.strftime("%Y%m01")))[0])["t2m"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")    
    dps=xr.open_dataset(glob.glob(str_fmt_sfc % ("2d",time.year,"2d",time.strftime("%Y%m01")))[0])["d2m"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")    
    us=xr.open_dataset(glob.glob(str_fmt_sfc % ("10u",time.year,"10u",time.strftime("%Y%m01")))[0])["u10"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")    
    vs=xr.open_dataset(glob.glob(str_fmt_sfc % ("10v",time.year,"10v",time.strftime("%Y%m01")))[0])["v10"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")    
    ps=xr.open_dataset(glob.glob(str_fmt_sfc % ("sp",time.year,"sp",time.strftime("%Y%m01")))[0])["sp"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")    
    zs=xr.open_dataset(glob.glob(str_fmt_sfc % ("z",time.year,"z",time.strftime("%Y%m01")))[0])["z"].\
            sel({"time":time.replace(minute=0)}).interp({"longitude":lon, "latitude":lat},method="nearest")   
    
    p=z.level
    dp=dewpoint_from_relative_humidity(t,r)
    prof_pres = np.flip(p.values.squeeze())
    prof_hgt = np.flip(z.values.squeeze()/9.8)
    prof_tmpc = np.flip(t.values.squeeze()-273.15)
    prof_dwpc = np.flip(np.array(dp).squeeze())
    prof_u = np.flip(u.values.squeeze()*1.94)
    prof_v = np.flip(v.values.squeeze()*1.94)
    agl_inds = prof_pres <= (ps.values/100.)
    prof = profile.create_profile(pres=np.insert(prof_pres[agl_inds], 0, ps.values/100.),
                                  hght=np.insert(prof_hgt[agl_inds], 0, zs.values/9.8),
                                  tmpc=np.insert(prof_tmpc[agl_inds], 0, ts.values-273.15),
                                  dwpc=np.insert(prof_dwpc[agl_inds], 0, dps.values-273.15),
                                  u=np.insert(prof_u[agl_inds], 0, us.values*1.94),
                                  v=np.insert(prof_v[agl_inds], 0, vs.values*1.94))
    sfcpcl = params.parcelx( prof, flag=1 )
    mupcl = params.parcelx( prof, flag=3 )
    mlpcl = params.parcelx( prof, flag=4 )    

    return prof, mupcl

def rotate_wind(prof, p_int):
    
    #From a profile (of u and v winds), rotate the wind profile so that Umean06 is westerly.
    
    u = interp.components(prof,p_int)[0].filled(np.nan)
    v = interp.components(prof,p_int)[1].filled(np.nan)

    x2,y2 = [1,0]
    x1,y1 = winds.mean_wind(prof,pbot=prof.pres.max(),ptop=interp.pres(prof,6000))
    
    angle = np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)

    u_prime = u*np.cos(angle) - v*np.sin(angle)
    v_prime = u*np.sin(angle) + v*np.cos(angle)

    ub = x2 / (np.sqrt(x2**2 + y2**2)) *1
    vb = y2 / (np.sqrt(x2**2 + y2**2)) *1    
    
    return u_prime, v_prime


if __name__ == "__main__":

	#Read daily maximum wind gusts at each station, created by resample_daily_max.py
	dmax_2p2km_point = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_2p2km.csv")
	dmax_obs = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_obs.csv")
	dmax_2p2km_point = dmax_2p2km_point.set_index(pd.to_datetime(dmax_2p2km_point.time))
	dmax_obs = dmax_obs.set_index(pd.to_datetime(dmax_obs.time))

	#Suspect gusts to drop - observations
	obs_times = pd.to_datetime(dmax_obs.time)
	dmax_obs.loc[(obs_times.dt.month == 12) & (obs_times.dt.year == 2010) & (dmax_obs.stn_id==55325) & (dmax_obs.gust >= 25),"gust"] = np.nan
	dmax_obs.loc[dmax_obs.gust>=129, "gust"] = np.nan 

	#Take the domain maximum for each day using daily maximum observations at each station with lightning
	dmax_2p2km_conv = dmax_2p2km_point.query("n_lightning_fl>=1").sort_values("wg10_2p2km_point").drop_duplicates("dt_floor_1D",keep="last").sort_index()
	dmax_obs_conv = dmax_obs.query("Lightning_observed_daily>=1").dropna(subset=["gust"]).sort_values("gust").drop_duplicates("dt_floor_1D",keep="last").sort_index()

	#For BARPA, calculate a composite sounding for each cluster
	for c in [1,2,0]:
		events_barpa = dmax_2p2km_conv[dmax_2p2km_conv.cluster==c][["time","stn_id","state"]]
		print("Cluster ",c," shape = ",events_barpa.shape)
		profs = []
		mu_pcl = []
		warnings.simplefilter("ignore")
		for name, row in tqdm.tqdm(events_barpa.iterrows()):
		    lat,lon=load_stn_info(row.state).query("stn_no=="+str(row.stn_id))[["lat","lon"]].values[0]
		    temp_prof, temp_mu_pcl = plot_sounding_barpa(
			pd.to_datetime(row.time).floor("6H"),lat,lon)
		    profs.append(temp_prof)
		    mu_pcl.append(temp_mu_pcl)        
			
		h_int = np.arange(250,17250,500)
		pres = np.nanmean(np.stack([interp.pres(p,h_int).filled(np.nan) for p in np.array(profs)]), axis=0)
		tmpc = np.nanmean(np.stack([interp.temp(p,pres).filled(np.nan) for p in np.array(profs)]), axis=0)
		dwpc = np.nanmean(np.stack([interp.dwpt(p,pres).filled(np.nan) for p in np.array(profs)]), axis=0)
		u = np.nanmean(np.stack([rotate_wind(p,pres)[0] for p in np.array(profs)]), axis=0)
		v = np.nanmean(np.stack([rotate_wind(p,pres)[1] for p in np.array(profs)]), axis=0)    
		pd.DataFrame({"pres":pres,"hgt":h_int,"tmpc":tmpc,"dwpc":dwpc,"u":u,"v":v}).\
			to_csv("/g/data/eg3/ab4502/ExtremeWind/composite_sounding_barpac_"+str(c)+".csv")
		pd.DataFrame({"mu_cape":[(p.bplus) for p in np.array(mu_pcl)],\
				"dcape":[params.dcape(p)[0] for p in profs]}).\
			to_csv("/g/data/eg3/ab4502/ExtremeWind/composite_sounding_mu_cape_barpac_"+str(c)+".csv")

	#Do the same for ERA5/obs
	for c in [1,2,0]:
		events_obs = dmax_obs_conv[dmax_obs_conv.cluster_era5==c][["time","stn_id","state"]]
		print("Cluster ",c," shape = ",events_obs.shape)
		profs = []
		mu_pcl = []
		warnings.simplefilter("ignore")
		for name, row in tqdm.tqdm(events_obs.iterrows()):
		    lat,lon=load_stn_info(row.state).query("stn_no=="+str(row.stn_id))[["lat","lon"]].values[0]
		    temp_prof, temp_mu_pcl = plot_sounding_era5(
			pd.to_datetime(row.time).floor("6H"),lat,lon)
		    profs.append(temp_prof)
		    mu_pcl.append(temp_mu_pcl)        
			
		h_int = np.arange(250,17250,500)
		pres = np.nanmean(np.stack([interp.pres(p,h_int).filled(np.nan) for p in np.array(profs)]), axis=0)
		tmpc = np.nanmean(np.stack([interp.temp(p,pres).filled(np.nan) for p in np.array(profs)]), axis=0)
		dwpc = np.nanmean(np.stack([interp.dwpt(p,pres).filled(np.nan) for p in np.array(profs)]), axis=0)
		u = np.nanmean(np.stack([rotate_wind(p,pres)[0] for p in np.array(profs)]), axis=0)
		v = np.nanmean(np.stack([rotate_wind(p,pres)[1] for p in np.array(profs)]), axis=0)    
		pd.DataFrame({"pres":pres,"hgt":h_int,"tmpc":tmpc,"dwpc":dwpc,"u":u,"v":v}).\
			to_csv("/g/data/eg3/ab4502/ExtremeWind/composite_sounding_era5_"+str(c)+".csv")
		pd.DataFrame({"mu_cape":[(p.bplus) for p in np.array(mu_pcl)],\
				"dcape":[params.dcape(p)[0] for p in profs]}).\
			to_csv("/g/data/eg3/ab4502/ExtremeWind/composite_sounding_mu_cape_era5_"+str(c)+".csv")
