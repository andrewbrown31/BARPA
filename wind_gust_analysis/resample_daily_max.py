import pandas as pd

def load_and_resample(state,interp):
    
    df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_"+state+interp+".csv").\
        dropna(subset="gust").\
        drop(columns=["time"]).\
        rename(columns={"Unnamed: 1":"time"})
    df = df.set_index(pd.to_datetime(df["time"]))
    df["state"] = state

    #BARPAC
    dmax_2p2km_point = df[["stn_id","wg10_2p2km_point","wgr_2p2km_point","n_lightning_fl","dt_floor_1D","bdsd","state","cluster","Umean06","lr13","qmean01","s06"]].\
			sort_values("wg10_2p2km_point").drop_duplicates(["stn_id","dt_floor_1D"],keep="last").sort_index()
	    
    #BARPA-R
    dmax_12km_point = df[["stn_id","wg10_12km_point","wgr_12km_point","n_lightning_fl","dt_floor_1D","state","cluster","Umean06","lr13","qmean01","s06"]].\
			sort_values("wg10_12km_point").drop_duplicates(["stn_id","dt_floor_1D"],keep="last").sort_index()    

    #ERAI
    dmax_erai_point = df[["stn_id","erai_wg10","Lightning_observed_daily","dt_floor_1D","state","cluster","Umean06","lr13","qmean01","s06"]].\
			sort_values("erai_wg10").drop_duplicates(["stn_id","dt_floor_1D"],keep="last").sort_index()    

    #OBS
    dmax_obs = df[["stn_id","gust","wgr_4","Lightning_observed_daily","bdsd_era5","dt_floor_1D","wg10_12km_rad","wg10_2p2km_rad","state","cluster_era5","Umean06_era5","lr13_era5","qmean01_era5","s06_era5"]].\
			sort_values("gust").drop_duplicates(["stn_id","dt_floor_1D"],keep="last").sort_index()    

    #Environment (bdsd)
    dmax_bdsd = df[["stn_id","bdsd_era5","dt_floor_1D","state","cluster_era5"]].\
			sort_values("bdsd_era5").drop_duplicates(["stn_id","dt_floor_1D"],keep="last").sort_index()   

    return dmax_2p2km_point, dmax_12km_point, dmax_obs, dmax_bdsd, dmax_erai_point

if __name__ == "__main__":

    interp = "_barpa_r_interp"    #interpolated to barpa-r
    #interp = ""		  #not interpolated

    dmax_2p2km_point = pd.DataFrame()
    dmax_12km_point = pd.DataFrame()
    dmax_obs = pd.DataFrame()
    dmax_bdsd = pd.DataFrame()
    dmax_erai_point = pd.DataFrame()
    for state in ["vic","nsw","sa","tas"]:
        print(state)
        temp_dmax_2p2km_point, temp_dmax_12km_point, temp_dmax_obs, temp_dmax_bdsd, temp_dmax_erai_point = load_and_resample(state,interp)
        dmax_2p2km_point = pd.concat([temp_dmax_2p2km_point, dmax_2p2km_point], axis=0)
        dmax_12km_point = pd.concat([temp_dmax_12km_point, dmax_12km_point], axis=0)
        dmax_obs = pd.concat([temp_dmax_obs, dmax_obs], axis=0)
        dmax_bdsd = pd.concat([temp_dmax_bdsd, dmax_bdsd], axis=0)
        dmax_erai_point = pd.concat([temp_dmax_erai_point, dmax_erai_point], axis=0)

    dmax_2p2km_point.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_2p2km"+interp+".csv")
    dmax_12km_point.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_12km"+interp+".csv")
    dmax_obs.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_obs"+interp+".csv")
    dmax_bdsd.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_bdsd"+interp+".csv")
    dmax_erai_point.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barpac_m_aws_dmax_erai"+interp+".csv")
