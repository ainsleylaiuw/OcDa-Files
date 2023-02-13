
from numpy import pi, sin, cos,pi
import plotly.express as px
import datetime
import plotly.graph_objects as go
from time import sleep
from tqdm import tqdm
from random import randint
from ooipy.request import hydrophone_request
from ooipy.tools import ooiplotlib as ooiplt
import os, json
import pandas as pd
import numpy as np
import datetime
import ooipy
import matplotlib.pyplot as plt
import matplotlib

from datetime import timedelta
import time
import warnings
warnings.filterwarnings("ignore")
from obspy import read
def choose_df(df,flag,verbose=True):
    
    ship_cols=['MMSI', 'SHIPNAME', 'VESSEL TYPE','SPEED (KNOTSx10)','COURSE', 'HEADING', 'TIMESTAMP UTC',
       'LENGTH', 'Year','ship_Loc','LAT','LON']
    if flag==1:
        cols=ship_cols + ['distance(in km) axial','axial_Loc']
        df=df[cols].rename(columns={'distance(in km) axial':'distance(in km)'})
    elif flag==2:
        cols=ship_cols + ['distance(in km) central cald','central_caldera_Loc']
        df=df[cols].rename(columns={'distance(in km) central cald':'distance(in km)'})
    elif flag==3:
        cols=ship_cols + ['distance(in km) eastern cald','eastern_caldera_Loc']
        df=df[cols].rename(columns={'distance(in km) eastern cald':'distance(in km)'})
    if verbose==True:
        print(' Max distance: {} and Min distance: {}'.format(df['distance(in km)'].max(),df['distance(in km)'].min()))
    return df

def choose_df_slope_hydrate(df,flag,verbose=True):
    ship_cols=['MMSI', 'VESSEL TYPE', 'TIMESTAMP UTC',
       'LENGTH','ship_Loc','LAT','LON','SOG','COG','Heading','VesselName']
    if flag==4:
        cols=ship_cols + ['distance(in km) oregon','oregon_slope_Loc']
        df=df[cols].rename(columns={'distance(in km) oregon':'distance(in km)'})
    elif flag==5:
        cols=ship_cols + ['distance(in km) hydrate','Southern_hydrate_Loc']
        df=df[cols].rename(columns={'distance(in km) hydrate':'distance(in km)'})
    if verbose==True:
        print(' Max distance: {} and Min distance: {}'.format(df['distance(in km)'].max(),df['distance(in km)'].min()))
    return df

def break_duration(isolated_ships,hydro):
    ncols=isolated_ships.shape[1]
    isolated_ships_2=pd.DataFrame()
    row={'MMSI':'','start_time':'','end_time':'','VESSEL TYPE':'','ud_group':''}
    idx=0
    for i in range(len(isolated_ships)):
        MMSI=isolated_ships['MMSI'].iloc[i]
        start=isolated_ships['start_time'].iloc[i]
        end=isolated_ships['end_time'].iloc[i]
        VESSEL_TYPE=isolated_ships['VESSEL TYPE'].iloc[i]
        ud_group=isolated_ships['ud_group'].iloc[i]
        duration=isolated_ships['len_of_recording'].iloc[i]
        str_start=start.strftime("%Y%m%d%H%M%S")
        str_end=end.strftime("%Y%m%d%H%M%S")
        row['MMSI']=MMSI
        row['VESSEL TYPE']=VESSEL_TYPE
        row['ud_group']=ud_group
        
        if duration <= 10:
            row['start_time']=start
            row['end_time']=end
            row['len_of_recording']=(end-start).total_seconds()/60
            print(row)
            row['instance_id']=hydro+'_'+start.strftime("%Y%m%d%H%M%S")+'_'+end.strftime("%Y%m%d%H%M%S")
            isolated_ships_2=isolated_ships_2.append(row,ignore_index=True)
            idx=idx+1
        else:
            temp_end=start+timedelta(minutes=10)
            temp_start=start
            while temp_end <= end:
                row['start_time']=temp_start
                row['end_time']=temp_end
                row['len_of_recording']=(temp_end-temp_start).total_seconds()/60
                row['instance_id']=hydro+'_'+temp_start.strftime("%Y%m%d%H%M%S")+'_'+temp_end.strftime("%Y%m%d%H%M%S")
                isolated_ships_2=isolated_ships_2.append(row,ignore_index=True)
                idx=idx+1

                temp_start=temp_end
                temp_end=temp_start+timedelta(minutes=10)
            if temp_start < end:
                row['start_time']=temp_start
                row['end_time']=end
                row['len_of_recording']=(end-temp_start).total_seconds()/60
                row['instance_id']=hydro+'_'+temp_start.strftime("%Y%m%d%H%M%S")+'_'+end.strftime("%Y%m%d%H%M%S")
                isolated_ships_2=isolated_ships_2.append(row,ignore_index=True)
                idx=idx+1
                
     
   #remove duration less than 10 minutes
    isolated_ships_2=isolated_ships_2[isolated_ships_2['len_of_recording']>=10]

        
        
    return isolated_ships_2


def load_saved_df_axial(hydrophone_idx,inner_rad,outer_rad):
    if hydrophone_idx==1:
        hydro='Axial_Base'
    elif hydrophone_idx==2:
        hydro='Central_Caldera'
    elif hydrophone_idx==3:
        hydro='Eastern_Caldera'
    suffix=str(inner_rad)+'_'+ str(outer_rad)+'.csv'
        
    isolated_ais=pd.read_csv(path_to_write+hydro+'/isolated_ais_'+ suffix)
    isolated_ais_10m=pd.read_csv(path_to_write+hydro+'/isolated_ais_10m_'+ suffix)
    
def load_saved_df_slope_hydrate(hydrophone_idx,inner_rad,outer_rad):
    
    if hydrophone_idx==1:
        hydro='Oregon_slope'
    elif hydrophone_idx==2:
        hydro='Southern_hydrate'
        
    suffix=str(inner_rad)+'_'+ str(outer_rad)+'.csv'
        
    isolated_ais=pd.read_csv(path_to_write+hydro+'/isolated_ais_'+ suffix)
    isolated_ais_10m=pd.read_csv(path_to_write+hydro+'/isolated_ais_10m_'+ suffix)
    
    
def get_isolated_ships(df,ud_vessel,rad,out_rad,min_d,verbose=True): #optimised version
  
    vessels=df[['MMSI','VESSEL TYPE']].drop_duplicates(subset=['MMSI'])
    ud_vessel_=ud_vessel[['vessel','ud_group']]
    df=df.sort_values(by=['TIMESTAMP UTC'],ascending=True)
    df_temp=df[df['distance(in km)']<(out_rad+1)] #find all ships inside radius = rad +1
    df_temp['rad_flag']=np.where(df_temp['distance(in km)'] <rad, 1, 0) #mark ships as 1 which are within radius=rad , otherwise 0
    df_temp['prev_MMSI']= df_temp['MMSI'].shift(1) # this will track when a new ship comes in
    df_temp['prev_rad_flag']= df_temp['rad_flag'].shift(1) # this will track when ship goes outside radius=rad
    count=0
    df_temp[['prev_MMSI','prev_rad_flag']].dropna(axis=0,inplace=True)
    #df_temp['break']=np.where((df_temp['MMSI']==df_temp['prev_MMSI']) & (df_temp['prev_rad_flag']==df_temp['rad_flag']),count,count+=1)
    df_temp['not_same_ship_as_prev']=~(df_temp['MMSI']==df_temp['prev_MMSI'])
    df_temp['xor_rad_flag_with_prev']=[bool(x)^bool(y) for x,y in zip(df_temp['rad_flag'],df_temp['prev_rad_flag'])]
    df_temp['or_ship_and_flag']=df_temp['not_same_ship_as_prev'] | df_temp['xor_rad_flag_with_prev']
    df_temp['break']=df_temp['or_ship_and_flag'].cumsum() # create partitions when a ship is continuously within radius=rad
    #df_temp_copy=df_temp.copy()
    #df_temp=df_temp.assign(flag=(df_temp['MMSI']==df_temp['prev_MMSI']) & (df_temp['prev_rad_flag']==df_temp['rad_flag']).cumsum())
    # df_temp.to_csv('ships_smoothpath.csv')
    df_temp=df_temp[df_temp['rad_flag']==1]
    df_temp_grp=df_temp.groupby(by=['MMSI','break']).agg({'TIMESTAMP UTC':['min','max','count']}).reset_index()
    df_temp_grp.columns=['MMSI','break','start_time','end_time','count']
    df_temp_grp.drop(df_temp_grp[df_temp_grp['end_time'] == df_temp_grp['start_time']].index,inplace=True)#remove ships which barely touched the radius
    #df_temp_grp.drop(df_temp_grp[df_temp_grp['count']<(min_t +1)].index,inplace=True) #filter ships with less than minimum number of timestamps
    df_temp_grp=df_temp_grp.merge(vessels,how='left',on='MMSI')
    
    df_temp_grp['len_of_recording']=(df_temp_grp['end_time']-df_temp_grp['start_time'])
    df_temp_grp['len_of_recording']=[x.total_seconds()/60.0 for x in df_temp_grp['len_of_recording']]
    df_temp_grp.drop(df_temp_grp[df_temp_grp['len_of_recording']<min_d].index,inplace=True)
    df_temp_grp.reset_index(inplace=True,drop=True)
    df_temp_grp.drop('break',axis=1,inplace=True)
    df_temp_grp=df_temp_grp.merge(ud_vessel_, how='left',left_on='VESSEL TYPE',right_on='vessel')
    df_temp_grp.drop(columns=['vessel'],axis=1,inplace=True)
    
    ##Merging additional ais data
    
    return df_temp_grp   

#Check status of ships' isolated timeframe in the overall ais dataset
def check_status_in_ais(ais,MMSI,min_time,max_time,rad):
    print('Min time : ', min_time)
    print('Max time : ', max_time)
    ais_test= ais[(ais['TIMESTAMP UTC']>min_time) & (ais['TIMESTAMP UTC']<max_time)]
    ais_test_lt=ais_test[ais_test['distance(in km)']<=rad]
    ais_test_gt=ais_test[ais_test['distance(in km)']>rad]
    n=ais_test_lt['MMSI'].nunique()
    min_distance_outside_rad=ais_test_gt['distance(in km)'].min()
    print('The number of unique isolated ships within the radius in the timeframe : ', n)
    print('The minimum distance of all other ships in the timeframe is : ',min_distance_outside_rad)
    return ais_test



def simp_spectrogram(hydrophone_idx,start_time,end_time,fmin=None,fmax=None):
    print('Start time : {} and End time : {}'.format(start_time,end_time))
    if hydrophone_idx==1:
        node='Axial_Base'
    elif hydrophone_idx==2:
        node='AXCC1'
    elif hydrophone_idx==3:
        node='AXEC2'
    elif hydrophone_idx==4:
        node='HYSB1'
    elif hydrophone_idx==5:
        node='HYS14'

    time_diff=end_time-start_time
    time_diff=time_diff.total_seconds()/60.0 
    if time_diff >10:
        end_time=start_time + timedelta(minutes=10)
    else:
        end_time=end_time
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node, verbose=True, zero_mean=True)
    print(data_trace)
    if data_trace==None:
        print('data trace is none. Continuing to next')
        pass
    else:
        spec = data_trace.compute_spectrogram(L = 256,avg_time=10, overlap=0.9)
        #spec.compute_psd_welch()

        print('/************************************************************************************************/')
        ooipy.plot(spec, fmin=fmin, fmax=fmax, xlabel_rot=30,vmax=110) #xlabel changed from 30 to 10
        #ooipy.tools.ooiplotlib.plot_spectrogram(spec)
        #plt.xlim([0, 10])
        plt.show()


def get_spectrogram_data(hydrophone_idx,start_time,end_time,fmin=None,fmax=None):
    if hydrophone_idx==1:
        node='Axial_Base'
    elif hydrophone_idx==2:
        node='AXCC1'
    elif hydrophone_idx==3:
        node='AXEC2'
    elif hydrophone_idx==4:
        node='HYSB1'
    elif hydrophone_idx==5:
        node='HYS14'
    
    time_diff=end_time-start_time
    time_diff=time_diff.total_seconds()/60.0 
    if time_diff >10:
        end_time=start_time + timedelta(minutes=10)
    else:
        end_time=end_time
    print(start_time)
    print(end_time)
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node, verbose=True, zero_mean=True)
    print(data_trace)
    if data_trace==None:
        print('data trace is none. Continuing to next')
        pass
    else:
        
        try:
            spec = data_trace.compute_spectrogram(L = 256,avg_time=10, overlap=0.9)
#             spec = data_trace.compute_spectrogram(L = 256,overlap=0.9)
        except TypeError:
            pass

        return spec
    

def save_spectrogram_acoustic_parallel(hydrophone_idx,i,data,inner_rad,outer_rad,path_to_write,fmin=None,fmax=None):
    if hydrophone_idx==1:
        node='Axial_Base'
        hydro_file='Axial_Base'
        hydro='AB'
    elif hydrophone_idx==2:
        node='AXCC1'
        hydro_file='Central_Caldera'
        hydro='CC'
    elif hydrophone_idx==3:
        node='AXEC2'
        hydro_file='Eastern_Caldera'
        hydro='EC'
    elif hydrophone_idx==4:
        node='HYSB1'
        hydro_file='Oregon_Slope'
        hydro='OS'
    elif hydrophone_idx==5:
        node='HYS14'
        hydro_file='Southern_Hydrate'
        hydro='SH'
        
        
        
    start_time=data.start_time.iloc[i]
    end_time=data.end_time.iloc[i]
    
    
    time_diff=end_time-start_time
    time_diff=time_diff.total_seconds()/60.0 
    if time_diff >10:
        end_time=start_time + timedelta(minutes=10)
    else:
        end_time=end_time
        
    str_starttime=start_time.strftime("%Y%m%d%H%M%S")
    str_endtime=end_time.strftime("%Y%m%d%H%M%S")
    spec_filename=path_to_write+hydro_file+'/Spectrograms'+'/'+hydro+'_'
    audio_filename=path_to_write+hydro_file+'/Audio'+'/'+hydro+'_'
    psd_filename=path_to_write+hydro_file+'/PSDs'+'/'+hydro+'_'
    mseed_filename=path_to_write+hydro_file+'/Mseed'+'/'+hydro+'_'
    psd_pickle_filename=path_to_write+hydro_file+'/PSD_pickles'+'/'+hydro+'_'
    
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node, verbose=True, zero_mean=True)
    
    
    if data_trace==None:
        print('data trace is none. Continuing to next')
        pass
    else:
        print('Saving wav files')
        
        data_trace.write(mseed_filename+str_starttime+'_'+str_endtime+'.mseed', format='MSEED')
        data_trace.wav_write(filename=audio_filename+str_starttime+'_'+str_endtime+'.wav')
        psd = data_trace.compute_psd_welch(L = 256, overlap=0.3)
        try:
            spectrogram = data_trace.compute_spectrogram(L = 256,avg_time=10, overlap=0.5)
            if (spectrogram==None) or (spectrogram.values.shape[0]<2) | (spectrogram.values.shape[1]<2) or (psd==None):
                # print('data trace is none. Continuing to next')
                pass
            else:
                ooiplt.plot_spectrogram(spectrogram,plot=False,save=True,fmin=fmin,fmax=fmax,vmax=110,filename=spec_filename+str_starttime+'_'+str_endtime+'.png')
                
                psd.save(filename=psd_pickle_filename+str_starttime+'_'+str_endtime+'.json')
                plot_psd(psd,plot=False,save=True,fmin=fmin,fmax=fmax,vmax=110,filename=psd_filename+str_starttime+'_'+str_endtime+'.png')
                
        except TypeError:
            pass


def get_acoustic(hydrophone_idx,start_time,end_time,fmin=None,fmax=None):
    #print('Start time : {} and End time : {}'.format(start_time,end_time))
    if hydrophone_idx==1:
        node='Axial_Base'
    elif hydrophone_idx==2:
        node='AXCC1'
    elif hydrophone_idx==3:
        node='AXEC2'
    elif hydrophone_idx==4:
        node='HYSB1'
    elif hydrophone_idx==5:
        node='HYS14'
    
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node, verbose=False, zero_mean=True)
    return data_trace

def get_acoustic_parallel(hydrophone_idx,i,data,fmin=None,fmax=None):
    #print('Start time : {} and End time : {}'.format(start_time,end_time))
    
    if hydrophone_idx==1:
        node='Axial_Base'
    elif hydrophone_idx==2:
        node='AXCC1'
    elif hydrophone_idx==3:
        node='AXEC2'
    elif hydrophone_idx==4:
        node='HYSB1'
    elif hydrophone_idx==5:
        node='HYS14'
        
    min_time=data.start_time[i]
    max_time=data.end_time[i]
    
    data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, node, verbose=False, zero_mean=True)
    return data_trace


def get_spectogram(hydrophone_idx,lone_ships,num=5,ideal_dur=10): #ideal_duration -> ideal duration in minutes
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('plotting of spectrogram started')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Total number of ships :',len(lone_ships))
    if len(lone_ships)<num:
        num=len(lone_ships)
    print('Displaying spectrogram for {} ships'.format(len(lone_ships)))
    for i in range(num):
        list_dur=lone_ships.iloc[i]
        min_time=list_dur.start_time
        min_time=datetime.datetime(min_time.year,min_time.month,min_time.day,min_time.hour,min_time.minute,0)
        #min_time=min_time.strftime('%Y-%m-%d %H:%M')
        #min_time=datetime.datetime.strptime(min_time,'%Y-%m-%d %H:%M')
        print(min_time)
        max_time=list_dur.end_time
        max_time=datetime.datetime(max_time.year,max_time.month,max_time.day,max_time.hour,max_time.minute,0)
        #max_time=max_time.strftime('%Y-%m-%d %H:%M')
        #max_time=datetime.datetime.strptime(max_time,'%Y-%m-%d %H:%M')
        start_time = min_time
        end_time=min_time+timedelta(minutes=ideal_dur)
        if end_time > max_time:
            end_time=max_time
        print('{}. MMSI: {} , VESSEL TYPE: {}'.format(i+1,lone_ships['MMSI'].iloc[i],lone_ships['VESSEL TYPE'].iloc[i]))
        print('Start time : {} and End time : {}'.format(start_time,end_time))
        simp_spectrogram(hydrophone_idx,start_time,end_time)
        #alter_spectrogram_func(start_time,end_time)
        #time.sleep(20)

def get_circle_coordinates(rad,lat,lon):
    R = rad #in meters

    center_lon = lon
    center_lat = lat

    t = np.linspace(0, 2*pi, 100)

    circle_lon =center_lon + R*cos(t)
    circle_lat =center_lat +  R*sin(t)


    coords=[]
    lat1=[]
    lon1=[]
    for lo, la in zip(list(circle_lon), list(circle_lat)):
        lat1.append(la)
        lon1.append(lo)
        coords.append([lo, la]) 

    ### Another way to find coordinates of a circle
    N = 100 # number of discrete sample points to be generated along the circle

    # generate points
    circlePoints = []
    lat=[]
    lon=[]
    for k in range(N):
        # compute
        angle = pi*2*k/N
        dx = R*cos(angle)
        dy = R*sin(angle)
        point = {}
        la=center_lat + (180/pi)*(dy/6378137)
        lo=center_lon + (180/pi)*(dx/6378137)/cos(center_lat*pi/180)
        lat.append(la)
        lon.append(lo)
        # add to list
        circlePoints.append([lo,la])

    return lat,lon    

def get_map_plot(fn,df,rad1,inner_rad2,outer_rad2,lat,lon,time=None):
    
    df1 = df[(df['distance(in km)']<rad1)]
    df2 = df[(df['distance(in km)']>inner_rad2) & (df['distance(in km)']<outer_rad2) ]
    df3=pd.concat([df1,df2])

    rad1=rad1 * 1000
    inner_rad2=inner_rad2*1000
    outer_rad2=outer_rad2*1000
    if fn=='scatter':

   
        fig = px.scatter_mapbox(df3,lat="LAT" ,lon="LON",mapbox_style='carto-positron',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],color = "VESSEL TYPE")
        
    elif fn=='density':
        fig = px.density_mapbox(df3,lat="LAT" ,lon="LON",mapbox_style='stamen-terrain',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE'],radius=3,
                color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]])
                    
      
    fig2 = px.scatter_mapbox(lat=[lat],lon=[lon],mapbox_style='carto-positron')
    # fig2.update_traces(marker_symbol='bus',selector=dict(type='scattermapbox'))
    fig2.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})



    lat_arr1,lon_arr1=get_circle_coordinates(rad1,lat,lon)
    lat_arr2,lon_arr2=get_circle_coordinates(inner_rad2,lat,lon)
    lat_arr3,lon_arr3=get_circle_coordinates(outer_rad2,lat,lon)


    fig4=px.scatter_mapbox(lat=lat_arr1,lon=lon_arr1,mapbox_style='carto-positron')
    fig4.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig5=px.scatter_mapbox(lat=lat_arr2,lon=lon_arr2,mapbox_style='carto-positron')
    fig5.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})

    fig6=px.scatter_mapbox(lat=lat_arr3,lon=lon_arr3,mapbox_style='carto-positron')
    fig6.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    fig.add_trace(fig6.data[0])
    #fig.add_trace(fig4.data[0])
    
    fig.update_layout(coloraxis_showscale=False,mapbox=dict(
       
                bearing=10,
                center=dict(
                    lat=lat,
                    lon=lon,
                )))
    
    fig.show()
    return df2

def get_isolated_map_plot(fn,df,rad1,inner_rad2,lat,lon,min_d):

    ais_test=pd.DataFrame()

    print('Finding isolated ships')
    iso_ships=get_isolated_ships(df,rad1,inner_rad2,min_d)
    print('Found isolated ships')

    print('Creating the dataframe with all ships and their locations in the isolated timeframes')
   
    
    for i in tqdm(range(len(iso_ships))):
        min_time=iso_ships.start_time[i]
        max_time=iso_ships.end_time[i]
        ais_test=pd.concat([ais_test,df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]])
    
    # df1 = df[(df['distance(in km) axial']<rad1)]
    # df2 = df[(df['distance(in km) axial']>inner_rad2) & (df['distance(in km) axial']<outer_rad2)]

    #ais_test= df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]
    rad1=rad1 * 1000
    inner_rad2=inner_rad2*1000
    print('Starting to plot now')
    if fn=='scatter':
        fig = px.scatter_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='carto-positron',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],color = "VESSEL TYPE")

    elif fn=='density':
        fig = px.density_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='stamen-terrain',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],radius=3,
                color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]])
    
    fig2 = px.scatter_mapbox(lat=[lat],lon=[lon],mapbox_style='carto-positron')
    # fig2.update_traces(marker_symbol='bus',selector=dict(type='scattermapbox'))
    fig2.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})



    lat_arr1,lon_arr1=get_circle_coordinates(rad1,lat,lon)
    lat_arr2,lon_arr2=get_circle_coordinates(inner_rad2,lat,lon)



    fig4=px.scatter_mapbox(lat=lat_arr1,lon=lon_arr1,mapbox_style='carto-positron')
    fig4.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig5=px.scatter_mapbox(lat=lat_arr2,lon=lon_arr2,mapbox_style='carto-positron')
    fig5.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})

    fig6 = px.line_mapbox( lat=[lat,lat_arr1[0]], lon=[lon,lon_arr1[0]],hover_name=[rad1,rad1])
    fig7 = px.line_mapbox( lat=[lat,lat_arr2[10]], lon=[lon,lon_arr2[10]],hover_name=[inner_rad2,inner_rad2])
    fig6.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig7.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig8 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr1[0]],
    lon = [lon,lon_arr1[0]],
    marker = {'size': 4,'color':'black'}))
    fig9 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr2[10]],
    lon = [lon,lon_arr2[10]],
    marker = {'size': 4,'color':'black'}))

    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    # fig.add_trace(fig6.data[0])
    # fig.add_trace(fig7.data[0])
    fig.add_trace(fig8.data[0])
    fig.add_trace(fig9.data[0])
    #fig.add_trace(fig4.data[0])
    
    fig.update_layout(coloraxis_showscale=False,mapbox=dict(
       
            bearing=0,
            center=dict(
                lat=lat,
                lon=lon,
            )))

    fig.show()

def get_single_isolated_map_plot(hydrophone_idx,fn,df,rad1,inner_rad2,lat,lon,min_d,fmin,fmax):

    

    print('Finding isolated ships')
    iso_ships=get_isolated_ships(df,rad1,inner_rad2,min_d)
    print('Found isolated ships')
    
   
    
    i=randint(0,len(iso_ships))
    print('showing for MMSI: ',iso_ships.MMSI[i])
    min_time=iso_ships.start_time[i]
    max_time=iso_ships.end_time[i]
    ais_test=df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]
    #check_status_in_ais(ziggly2.MMSI[i],min_time,max_time,rad1)
    simp_spectrogram(hydrophone_idx,min_time,max_time,fmin,fmax)
    
    # df1 = df[(df['distance(in km) axial']<rad1)]
    # df2 = df[(df['distance(in km) axial']>inner_rad2) & (df['distance(in km) axial']<outer_rad2)]

    #ais_test= df[(df['TIMESTAMP UTC']>=min_time) & (df['TIMESTAMP UTC']<=max_time)]
    rad1=rad1 * 1000
    inner_rad2=inner_rad2*1000
    print('Starting to plot now')
    if fn=='scatter':
        fig = px.scatter_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='carto-positron',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],color = "VESSEL TYPE")

    elif fn=='density':
        fig = px.density_mapbox(ais_test,lat="LAT" ,lon="LON",mapbox_style='stamen-terrain',hover_name="TIMESTAMP UTC",
                    hover_data= ['SPEED (KNOTSx10)','VESSEL TYPE','distance(in km)'],radius=3,
                color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]])
    
    fig2 = px.scatter_mapbox(lat=[lat],lon=[lon],mapbox_style='carto-positron')
    # fig2.update_traces(marker_symbol='bus',selector=dict(type='scattermapbox'))
    fig2.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})



    lat_arr1,lon_arr1=get_circle_coordinates(rad1,lat,lon)
    lat_arr2,lon_arr2=get_circle_coordinates(inner_rad2,lat,lon)



    fig4=px.scatter_mapbox(lat=lat_arr1,lon=lon_arr1,mapbox_style='carto-positron')
    fig4.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})


    fig5=px.scatter_mapbox(lat=lat_arr2,lon=lon_arr2,mapbox_style='carto-positron')
    fig5.update_traces(marker = {'size': 4, 'color':'black','opacity':0.9})

    fig6 = px.line_mapbox( lat=[lat,lat_arr1[0]], lon=[lon,lon_arr1[0]],hover_name=[rad1,rad1])
    fig7 = px.line_mapbox( lat=[lat,lat_arr2[10]], lon=[lon,lon_arr2[10]],hover_name=[inner_rad2,inner_rad2])
    fig6.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig7.update_traces(marker = {'size': 8, 'color':'black','opacity':0.9})
    fig8 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr1[0]],
    lon = [lon,lon_arr1[0]],
    marker = {'size': 4,'color':'black'}))
    fig9 = go.Figure(go.Scattermapbox(
    mode = "markers+lines",
    lat = [lat,lat_arr2[10]],
    lon = [lon,lon_arr2[10]],
    marker = {'size': 4,'color':'black'}))

    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    # fig.add_trace(fig6.data[0])
    # fig.add_trace(fig7.data[0])
    fig.add_trace(fig8.data[0])
    fig.add_trace(fig9.data[0])
    #fig.add_trace(fig4.data[0])
    
    fig.update_layout(coloraxis_showscale=False,mapbox=dict(
       
            bearing=0,
            center=dict(
                lat=lat,
                lon=lon,
            )))

    fig.show()

def isolated_ais(ais,iso_ships,ud_vessel,inner_rad):
    data=pd.DataFrame()
    ud_vessel_=ud_vessel[['vessel','ud_group']]
    
    for i in tqdm(range(len(iso_ships))):
        min_time=iso_ships.start_time.iloc[i]
        max_time=iso_ships.end_time.iloc[i]
        idx=iso_ships.instance_id.iloc[i]
        temp=ais[(ais['TIMESTAMP UTC']>=min_time) & (ais['TIMESTAMP UTC']<=max_time) & (ais['distance(in km)']<inner_rad)]
        if len(temp)==0:
            temp=data.iloc[[i-1]]
        temp['instance_id']=idx
        data=pd.concat([data,temp])
        
    data=data.merge(ud_vessel_, how='left',left_on='VESSEL TYPE',right_on='vessel') 
    data.drop(columns=['vessel'],inplace=True)
    return data

def ais_ping_distribution(ais,n=10,hist_show=False,bar_show=False):
    ais = ais.sort_values(by=['MMSI','TIMESTAMP UTC'],ascending=True)
    ais['prev TIMESTAMP UTC'] = ais.groupby(by='MMSI')['TIMESTAMP UTC'].shift(1)
    ais_temp=ais.copy()
    ais_temp.dropna(axis=0,how='any',subset=['prev TIMESTAMP UTC'],inplace=True)
    ais_temp['ping_time']=(ais_temp['TIMESTAMP UTC'] - ais['prev TIMESTAMP UTC']).dt.total_seconds()/60
    ships_pings=ais_temp.groupby(by=['VESSEL TYPE', 'MMSI']).agg({'ping_time': ['mean','min','max','median']}).reset_index()
    ships_pings.columns=['VESSEL TYPE','MMSI','mean_ping_time','min_ping_time','max_ping_time','median_ping_time']
    vessels_pings=ais_temp.groupby(by=['VESSEL TYPE']).agg({'MMSI': pd.Series.nunique,'ping_time': ['mean','min','max','median']}).reset_index()
    vessels_pings.columns=['VESSEL TYPE','distinct count ships','mean_ping_time','min_ping_time','max_ping_time','median_ping_time']
    vessels_pings=vessels_pings.sort_values(by='distinct count ships',ascending=False)

    if(hist_show):
        for vessel in vessels_pings['VESSEL TYPE'].unique()[:n]:
            fig,ax= plt.subplots(2,2,figsize=(10,10))
            fig.suptitle(' Distribution for {} vessel'.format(vessel), fontsize=20)
            ax[0,0].hist(ships_pings['mean_ping_time'][ships_pings['VESSEL TYPE']==vessel],bins=10)
            ax[0,1].hist(ships_pings['min_ping_time'][ships_pings['VESSEL TYPE']==vessel],bins=10)
            ax[1,0].hist(ships_pings['max_ping_time'][ships_pings['VESSEL TYPE']==vessel],bins=10)
            ax[1,1].hist(ships_pings['median_ping_time'][ships_pings['VESSEL TYPE']==vessel],bins=10)

    if(bar_show):
        fig,ax= plt.subplots(2,2,figsize=(20,20))
        fig.suptitle(' Successive ping gaps by vessel type ', fontsize=20)
        ax[0,0].bar(vessels_pings['VESSEL TYPE'].iloc[:n],vessels_pings['mean_ping_time'].iloc[:n])
        ax[0,0].set_title('Mean')
        ax[0,0].set_xticklabels(vessels_pings['VESSEL TYPE'].unique()[:n],rotation = 45)

        ax[0,1].bar(vessels_pings['VESSEL TYPE'].iloc[:n],vessels_pings['min_ping_time'].iloc[:n])
        ax[0,1].set_title('Min')
        ax[0,1].set_xticklabels(vessels_pings['VESSEL TYPE'].unique()[:n],rotation = 45)

        ax[1,0].bar(vessels_pings['VESSEL TYPE'].iloc[:n],vessels_pings['max_ping_time'].iloc[:n])
        ax[1,0].set_title('Max')
        ax[1,0].set_xticklabels(vessels_pings['VESSEL TYPE'].unique()[:n],rotation = 45)

        ax[1,1].bar(vessels_pings['VESSEL TYPE'].iloc[:n],vessels_pings['median_ping_time'].iloc[:n])
        ax[1,1].set_title('Median')
        ax[1,1].set_xticklabels(vessels_pings['VESSEL TYPE'].unique()[:n],rotation = 45)

       

    return ships_pings,vessels_pings




def extend_ais(hydrophone_idx,shared_link,local_link):
    if hydrophone_idx==1:
        hydro_file='Axial_Base'
    elif hydrophone_idx==2:
        hydro_file='Central_Caldera'
    elif hydrophone_idx==3:
        hydro_file='Eastern_Caldera'
    elif hydrophone_idx==4:
        hydro_file='Oregon_Slope'
    elif hydrophone_idx==5:
        hydro_file='Southern_Hydrate'
        
    df_spec=pd.DataFrame()
    df_audio=pd.DataFrame()
    df_mseed=pd.DataFrame()
    df_psds=pd.DataFrame()
    df_psdpickle=pd.DataFrame()
    
    dict_spec={'instance_id':'','spectrogram':''}
    dict_audio={'instance_id':'','audio':''}
    dict_mseed={'instance_id':'','mseed':''}
    dict_psds={'instance_id':'','psd':''}
    dict_psdpickle={'instance_id':'','psd_pickle':''}
    
    #spectrogram
    arr_spec=os.listdir(local_link+hydro_file+'/Spectrograms/')
    for name in arr_spec:
        
        file=name.split('.')[0]
        new_file=file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2]
#         dst = local_link+hydro_file+'/Spectrograms/'+new_file+ '.jpg'
#         src =local_link+hydro_file+'/Spectrograms/'+name # foldername/filename, if .py file is outside folder
#         os.rename(src, dst)
        
        dict_spec['instance_id']=file
        dict_spec['spectrogram']=shared_link+'/Ships/'+hydro_file+'/Spectrograms/'+new_file+ '.png'
        temp=pd.DataFrame([dict_spec])
        df_spec=pd.concat([df_spec,temp])
        

    #Audio
    arr_audio=os.listdir(local_link+hydro_file+'/Audio/')
    for name in arr_audio:
        
        file=name.split('.')[0]
        new_file=file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2]
#         dst = local_link+hydro_file+'/Audio/'+new_file+ '.wav'
#         src =local_link+hydro_file+'/Audio/'+name # foldername/filename, if .py file is outside folder
#         os.rename(src, dst)
        
        dict_audio['instance_id']=file
        dict_audio['audio']=shared_link+'/Ships/'+hydro_file+'/Audio/'+new_file+ '.wav'
        temp=pd.DataFrame([dict_audio])
        df_audio=pd.concat([df_audio,temp])
    
    #Mseed
    arr_mseed=os.listdir(local_link+hydro_file+'/Mseed/')
    for name in arr_mseed:
        file=name.split('.')[0]
        new_file=file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2]
#         dst = local_link+hydro_file+'/Mseed/'+new_file+ '.mseed'
#         src =local_link+hydro_file+'/Mseed/'+name # foldername/filename, if .py file is outside folder
#         os.rename(src, dst)
        
        dict_mseed['instance_id']=file
        dict_mseed['mseed']=shared_link+'/Ships/'+hydro_file+'/Mseed/'+new_file+ '.mseed'
        temp=pd.DataFrame([dict_mseed])
        df_mseed=pd.concat([df_mseed,temp])
    
    #PSDs
    arr_psds=os.listdir(local_link+hydro_file+'/PSDs/')
    for name in arr_psds:
        
        file=name.split('.')[0]
        new_file=file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2]
#         dst = local_link+hydro_file+'/PSDs/'+new_file+ '.jpg'
#         src =local_link+hydro_file+'/PSDs/'+name # foldername/filename, if .py file is outside folder
#         os.rename(src, dst)
        
        dict_psds['instance_id']=file
        dict_psds['psd']=shared_link+'/Ships/'+hydro_file+'/PSDs/'+new_file+ '.png'
        temp=pd.DataFrame([dict_psds])
        df_psds=pd.concat([df_psds,temp])
    
    #PSD pickles
    arr_psdpickle=os.listdir(local_link+hydro_file+'/PSD_pickles/')
    for name in arr_psdpickle:
        
        file=name.split('.')[0]
        new_file=file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2]
#         dst = local_link+hydro_file+'/PSD_pickles/'+new_file+ '.pkl'
#         src =local_link+hydro_file+'/PSD_pickles/'+name # foldername/filename, if .py file is outside folder
#         os.rename(src, dst)
        
        dict_psdpickle['instance_id']=file
        dict_psdpickle['psd_pickle']=shared_link+'/Ships/'+hydro_file+'/PSD_pickles/'+new_file+ '.pkl'
        temp=pd.DataFrame([dict_psdpickle])
        df_psdpickle=pd.concat([df_psdpickle,temp])
    
    return df_spec,df_audio,df_mseed,df_psds,df_psdpickle


def save_spectrogram_noship(i,hydrophone_idx,fmin,fmax,rad,hydro_file,hydro,df_time_grp):
    
    min_time=df_time_grp.start_time.iloc[i]
    max_time=df_time_grp.end_time.iloc[i]
    str_starttime=min_time.strftime("%Y%m%d%H%M%S")
    str_endtime=max_time.strftime("%Y%m%d%H%M%S")
    df_time_grp['instance_id'].iloc[i]=hydro+'_'+str_starttime+'_'+str_endtime+'_'+str(i)
    spectrogram=get_spectrogram_data(hydrophone_idx,min_time,max_time,fmin=fmin,fmax=fmax)
    if (spectrogram==None):
        print('data trace is none. Continuing to next')
        pass
    else:
        if (spectrogram.values.shape[0]<2) | (spectrogram.values.shape[1]<2):
            pass
        else:
            print(path_to_write)
            #spectrogram.visualize(save_spec=True,plot_spec=False,filename=path_to_write+hydro+'/Spectrogram'+'/isolated_ships_'+str(inner_rad)+'_'+ str(outer_rad)+'_'+str(i)+'.png')
            ooiplt.plot_spectrogram(spectrogram,plot=False,save=True,vmax=110,fmin=fmin,fmax=fmax,filename=path_to_write+hydro_file+'/Spectrograms'+'/'+hydro+'_'+str_starttime+'_'+str_endtime+'_'+str(i)+'.png')
            
            
            
def plot_psd(psd_obj, **kwargs):
    """
    Plot a :class:`ooipy.hydrophone.basic.Psd` object using the
    matplotlib package.
    Parameters
    ----------
    spec_obj : :class:`ooipy.hydrophone.basic.Psd`
        Psd object to be plotted
    kwargs :
        See matplotlib doccumentation for list of arguments. Additional
        arguments are
        * plot : bool
            If False, figure will be closed. Can save time if only
            saving but not plotting is desired. Default is True
        * save : bool
            If True, figure will be saved under **filename**. Default is
            False
        * new_fig : bool
            If True, matplotlib will create a new fugure. Default is
            True
        * filename : str
            filename of figure if saved. Default is "spectrogram.png"
        * xlabel_rot : int or float
            rotation angle (deg) of x-labels. Default is 70
        * fmin : int or float
            minimum frequency. Default is 0
        * fmax : int or float
            maximum frequency. Default is 32000
        * vmin : int or float
            lower limit of level axis (colormap). Default is 20
        * vmax : int or float
            upper limit of level axis (colormap). Default is 80
        * figsize : (int, int)
            width and height of figure. Default is (16, 9)
        * dpi : int
            dots per inch, passed to matplotlib figure.savefig()
        * fontsize : int
            fontsize of saved plot, passed to matplotlib figure
    """

    # check for keys
    if "plot" not in kwargs:
        kwargs["plot"] = True
    if "save" not in kwargs:
        kwargs["save"] = False
    if "new_fig" not in kwargs:
        kwargs["new_fig"] = True
    if "filename" not in kwargs:
        kwargs["filename"] = "psd.png"
    if "title" not in kwargs:
        kwargs["title"] = "PSD"
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "frequency"
    if "xlabel_rot" not in kwargs:
        kwargs["xlabel_rot"] = 0
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "spectral level"
    if "fmin" not in kwargs:
        kwargs["fmin"] = 0.0
    if "fmax" not in kwargs:
        kwargs["fmax"] = 32000.0
    if "vmin" not in kwargs:
        kwargs["vmin"] = 20.0
    if "vmax" not in kwargs:
        kwargs["vmax"] = 80.0
    if "figsize" not in kwargs:
        kwargs["figsize"] = (16, 9)
    if "dpi" not in kwargs:
        kwargs["dpi"] = 100
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = 22

    # set backend for plotting/saving:
    if not kwargs["plot"]:
        matplotlib.use("Agg")
    font = {"size": kwargs["fontsize"]}
    matplotlib.rc("font", **font)

    if len(psd_obj.freq) != len(psd_obj.values):
        f = np.linspace(0, len(psd_obj.values) - 1, len(psd_obj.values))
    else:
        f = psd_obj.freq

    # plot PSD object
    if kwargs["new_fig"]:
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        ax.set_xscale('log') # new addition
    plt.plot(f, psd_obj.values)
    plt.ylabel(kwargs["ylabel"])
    plt.xlabel(kwargs["xlabel"])
    plt.xlim([kwargs["fmin"], kwargs["fmax"]])
    plt.ylim([kwargs["vmin"], kwargs["vmax"]])
    plt.xticks(rotation=kwargs["xlabel_rot"])
    plt.title(kwargs["title"])
    plt.grid(True)

    if kwargs["save"]:
        plt.savefig(kwargs["filename"], bbox_inches="tight", dpi=kwargs["dpi"])

    if not kwargs["plot"]:
        plt.close(fig)

