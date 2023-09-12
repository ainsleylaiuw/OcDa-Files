# data
from obspy import read as obsread
import ooipy
from ooipy.tools import ooiplotlib as ooiplt
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg

# IO
from io import BytesIO
import io
import json
from scipy.io import wavfile

# general
from datetime import datetime
import re

from scipy.signal import welch

from geopy.distance import geodesic
import plotly.express as px
import datetime

CC_node_LF = 'Central_Caldera'
EC_node_LF = 'Eastern_Caldera'
ecdist = (45.9399, -129.9742)
ccdist = (45.9549, -130.0089)
abdist = (45.8168, -129.7543)
def animate_map(ais, starttime, endtime):
    """
    Code taken from Animation Map.ipynb
    Plots an interactive map that displays the positions of MMSIs in a given AIS dataframe for the given timeframe
    subais.loc[] lines have a certain format that matches up with the given AIS dataframe format so if we change columns in that, we must change them accordingly in this function
    """
    subais = ais[(ais['TIMESTAMP UTC'] > starttime) & (ais['TIMESTAMP UTC'] < endtime)]
    time = subais['TIMESTAMP UTC']
    for i in time:
        #subais.loc[len(subais.index)] = [0,0,'cc hydrophone'+str(i),'cc hydrophone',0,0,0, -130.0089,45.9549,0,0, i, 0,0]
        #subais.loc[len(subais.index)] = [0,0, 'cc hydrophone '+str(i), 'cc hydrophone', 0,0,0,i,0, 0,0,45.9549, -130.0089, 0, 0,0,0,0,0,0]
        subais.loc[len(subais.index)] = [0, 'cc hydrophone '+str(i), 'cc hydrophone', 0, 0, 45.9549, -130.0089, 0, 0, i, 0, 0, 0, 0]
        
        #subais.loc[len(subais.index)] = [0,0,'ec hydrophone'+str(i),'ec hydrophone',0,0,10, -129.9742,45.9399,0,0, i, 0,0]
        #subais.loc[len(subais.index)] = [0,0, 'ec hydrophone '+str(i), 'ec hydrophone', 0,0,0,i,0, 0,0,45.9399, -129.9742, 0, 0,0,0,0,0,0]
        subais.loc[len(subais.index)] = [0, 'ec hydrophone '+str(i), 'ec hydrophone', 0, 0, 45.9399, -129.9742, 0, 0, i, 0, 0, 0, 0]

        #subais.loc[len(subais.index)] = [0,0, 'ab hydrophone '+str(i), 'ab hydrophone', 0,0,0,i,0, 0,0,45.8168, -129.7543, 0, 0,0,0,0,0,0]
        subais.loc[len(subais.index)] = [0, 'ab hydrophone '+str(i), 'ab hydrophone', 0, 0, 45.8168, -129.7543, 0, 0, i, 0, 0, 0, 0]
    subais = subais.reset_index()
    for i in range(len(subais)):
        subais.loc[i,'TIMESTAMP UTC'] = subais.iloc[i]['TIMESTAMP UTC']- datetime.timedelta(minutes=subais.iloc[i]['TIMESTAMP UTC'].minute % 10)
    subais['TIMESTAMP UTC'] = subais['TIMESTAMP UTC'].dt.to_period('T')
    fig = px.scatter_mapbox(subais,
              lat="LAT" ,
              lon="LON",
              hover_name="VESSEL TYPE",
              hover_data= ['SPEED (KNOTSx10)','ccdist'],
              color="SHIPNAME",
              animation_frame=subais['TIMESTAMP UTC'].astype(dtype=str),
              mapbox_style='carto-positron',                 
              zoom=5)
    fig.show()

def get_PSD_dict(meta_data_CSV_path, location, close_distance=False):
    """
    Generate a dictionary that holds PSD data
    Ex. psd_dict[ship_type][freq_values] # not actual key names
    This data gets passed into the spdf generation and plotting functions
    
    Parameters
    ----------
    meta_data_CSV_path : str
        A file path that points to the .csv that contains the metadata. 
        Mainly used to group based on ship type (ud_group). Also matches instance_id
        to corresponding ship type.

    location : str
        String containing location of the hydrophone (i.e. Axial_Base).
        Used for file pathing

    Returns
    -------
    dict: a nested dictionary for ship type and their respective PSD data
    """

    df = pd.read_csv(meta_data_CSV_path, sep=',')

    if close_distance:
        depth_dict = {
            "Axial_Base": 3,
            "Eastern_Caldera": 1.5,
            "Central_Caldera": 1.5,
            "Oregon_Slope": 3,
            "Southern_Hydrate": 0.8
        }
        # subset to close instances
        df = df.loc[df["mean_distance(in kms)"] <= depth_dict[location]]
        df = df.reset_index(drop=True)

    # EC is the only one with missing ship type and only 4 of them, just ignore those
    # .dropna() followed by .groupby() cleans and groups
    grouped_df = df.dropna(subset=['ud_group']).groupby('ud_group')

    # https://stackoverflow.com/questions/27405483/how-to-loop-over-grouped-pandas-dataframe
    PSDs = {}
    for group_name, df_group in grouped_df:
        PSD_obj_list = []
        PSD_freq = []
        PSD_val = []
        dictPSD = {'objects': PSD_obj_list, 'frequencies': PSD_freq, 'values': PSD_val}

        # for-loop to compile list of PSD data objects pulled from column
        for row_index, row in df_group.iterrows():
            inst_id = df['instance_id'].iloc[row_index]
            # get pickle files (DL locally? Call some API?)
            # local storage method
            data_path = 'data/' + location + '/'
            pklfilepath = data_path + 'PSD_pickles/' + inst_id + '.pkl'
            # try-except deals with missing pickles
            try:
                with open(pklfilepath, 'rb') as f:
                    dict1 = json.load(f)
                #print("----------PKL FOUND----------")
                # create PSD object and fill out PSD object fields
                psd = ooipy.Psd(dict1['f'], dict1['psd'])
                dictPSD['objects'].append(psd)
                dictPSD['frequencies'].append(psd.freq)
                dictPSD['values'].append(psd.values)
            except FileNotFoundError:
                #print("Could not find pkl")
                pass
        PSDs.update({group_name: dictPSD})
    return PSDs
  
def get_spdf(spec, fs_hz=200, fmax=None, spl_bins=np.linspace(0, 120, 481),
             percentiles=[1, 5, 10, 50, 90, 95, 99]):
    """
    function pulled from https://github.com/Ocean-Data-Lab/Website-backend
    """
    if fmax is None:
        fmax = 100 #spec.frequency[-1]

    n_freq_bin = int(len(spec['frequencies'][0]) * fmax/(fs_hz/2)) + 1

    spdf_dct = {'freq': np.array(np.linspace(0, fmax, n_freq_bin)),
                'spl': spl_bins[:-1],
                'pdf': np.empty((n_freq_bin, 480))
                #'number_psd': len(spec.time) this isn't used
                }

    for p in percentiles:
        spdf_dct[str(p)] = np.empty(n_freq_bin)

    for idx, freq_bin in enumerate(np.asarray(spec['values']).T[:n_freq_bin - 1]):
        hist, _ = np.histogram(freq_bin, bins=spl_bins, density=True)
        spdf_dct['pdf'][idx] = hist
        spdf_dct['50'][idx] = np.median(freq_bin)
        for p in percentiles:
            spdf_dct[str(p)][idx] = np.nanquantile(freq_bin, p/100)

    return spdf_dct

def plot_spdf(spdf, vmin=0.003, vmax=0.2, vdelta=0.0025, save=False, filename=None, log=True, title='Spectral PDF', shipType=None, location=None):
    """
    function pulled from https://github.com/Ocean-Data-Lab/Website-backend
    """
    cbarticks = np.arange(vmin, vmax+vdelta, vdelta)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.contourf(spdf['freq'], spdf['spl'], np.transpose(spdf['pdf']),
                     cbarticks, norm=colors.Normalize(vmin=vmin, vmax=vmax),
                     cmap='jet', extend='max', alpha=0.50, linewidth=0)

    # plot some percentiles:
    plt.plot(spdf['freq'], spdf['1'], color='black')
    plt.plot(spdf['freq'], spdf['5'], color='black')
    plt.plot(spdf['freq'], spdf['10'], color='black')
    plt.plot(spdf['freq'], spdf['50'], color='black')
    plt.plot(spdf['freq'], spdf['90'], color='black')
    plt.plot(spdf['freq'], spdf['95'], color='black')
    plt.plot(spdf['freq'], spdf['99'], color='black')

    plt.ylabel(r'spectral level (dB rel $1 \mathrm{\frac{μ Pa^2}{Hz}}$)')
    plt.xlabel('frequency (Hz)')
    plt.ylim([36, 100])
    plt.xlim([0, 90])
    if log:
        plt.xscale('log')

    plt.colorbar(im, ax=ax, ticks=[vmin, vmin + (vmax-vmin)/4, vmin + (vmax-vmin)/2,
                 vmin + 3*(vmax-vmin)/4,  vmax],  pad=0.03, label='probability', format='%.3f')
    plt.tick_params(axis='y')
    plt.grid(True)
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D(
        [0], [0], label='percentiles: 1, 5, 10, 50, 90, 95, 99', color='k')
    handles.extend([line])
    plt.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    plt.xlim((200/512, 100))
    filepath = 'data/'+location+'/'
    #plt.savefig(filepath+shipType+'.jpg', format='jpg')
    plt.savefig(filepath+title+'.jpg', format='jpg')
    plt.close() # suppress plotting

def get_PSD_dict_given_filt_df(df, location, close_distance=False):
    df = df.reset_index(drop=True)
    """df = pd.read_csv(meta_data_CSV_path, sep=',')

    if close_distance:
        depth_dict = {
            "Axial_Base": 3,
            "Eastern_Caldera": 1.5,
            "Central_Caldera": 1.5,
            "Oregon_Slope": 3,
            "Southern_Hydrate": 0.8
        }
        # subset to close instances
        df = df.loc[df["mean_distance(in kms)"] <= depth_dict[location]]
        df = df.reset_index(drop=True)"""

    # https://stackoverflow.com/questions/27405483/how-to-loop-over-grouped-pandas-dataframe
    PSD_obj_list = []
    PSD_freq = []
    PSD_val = []
    dictPSD = {'objects': PSD_obj_list, 'frequencies': PSD_freq, 'values': PSD_val}

    # for-loop to compile list of PSD data objects pulled from column
    for row_index, row in df.iterrows():
        inst_id = df['instance_id'].iloc[row_index]
        # get pickle files (DL locally? Call some API?)
        # local storage method
        data_path = 'data/' + location + '/'
        pklfilepath = data_path + 'PSD_pickles/' + inst_id + '.json'
        #if (location == 'Eastern_Caldera'):
        #    pklfilepath = data_path + 'PSD_pickles/' + inst_id + '.pkl'
        # try-except deals with missing pickles
        try:
            print(pklfilepath)
            with open(pklfilepath, 'rb') as f:
                dict1 = json.load(f)
            #print("----------PKL FOUND----------")
            # create PSD object and fill out PSD object fields
            psd = ooipy.Psd(dict1['f'], dict1['psd'])
            dictPSD['objects'].append(psd)
            dictPSD['frequencies'].append(psd.freq)
            dictPSD['values'].append(psd.values)
        except FileNotFoundError:
            #print("Could not find pkl")
            #pass
            try:
                pklfilepath = data_path + 'PSD_pickles/' + inst_id + '.pkl'
                with open(pklfilepath, 'rb') as f:
                    dict1 = json.load(f)
                #print("----------PKL FOUND----------")
                # create PSD object and fill out PSD object fields
                psd = ooipy.Psd(dict1['f'], dict1['psd'])
                dictPSD['objects'].append(psd)
                dictPSD['frequencies'].append(psd.freq)
                dictPSD['values'].append(psd.values)
            except FileNotFoundError:
                pass
    #PSDs.update({group_name: dictPSD})
    return dictPSD

def Rx_level_from_df(df,location):
    """
    Requires dataframe for specific MMSI and maybe filtered to be sufficiently close to hydrophone.

    Return lists that contain range values (closest to farthest) within a set range (df param should already be filtered),
    and a list that contains the receive level, which is calculated from the FFT of the time series data.
    Due to needing the time series data, may be time-consuming to run.
    This data can be used to get source levels by adding/making up from Tx loss using a separate function.

    Calculation for Receive Level/Sound Exposure is found in the Guerra, Thode et al. paper.

    Parameters
    ----------
    df : Pandas dataframe
        A dataframe that is filtered to only contain one MMSI. Should probably also be close to hydrophone.

    Returns
    -------
    R : a list of range values in kilometers. When using it for Tx loss, need to make it an array and *1000 to get meters.

    Rx_level : a list of receive levels
    """

    df = df.sort_values(by=['distance(in km)'])
    df = df.reset_index(drop=True)

    Rx_level = []
    R = []
    file404 = 0
    for row_index, row in df.iterrows():
        try:
            inst = df['instance_id'].loc[row_index]
            #print(inst)
            st = obsread('data/'+location+'/Mseed/'+inst+'.mseed')
            data = st[-1].data
            #print(st[-1])
            
            Fs = 200
            N = len(data)
            freqs = np.linspace(-(Fs/2), Fs/2, N) # divide by 2 for nyquist, center on zero for fftshift. N points to match with N points in recording
            ft_data = np.fft.fft(data)
            idx_5hz = 63000 # index corresponding to frequency of 5 on the frequency domain centered on zero
            idx_90hz = 114000

            # Follows equation to get sound exposure
            # Guerra_Thode_et al_2011
            SE = np.sum(np.fft.fftshift(np.abs(ft_data)**2)[idx_5hz:idx_90hz] / (N * Fs))
            SEL = 10 * np.log10(SE) # dB conversion
            #print(f'SEL = {SEL}')
            Rx_level.append(SEL)
            R.append(df['distance(in km)'].loc[row_index])
            #print(df['mean_distance(in kms)'].loc[row_index])
        except FileNotFoundError:
            file404 +=  1
            pass

    print(f'Files not found: {file404}')
    return R, Rx_level

def Rx_level_from_df_SPEED(df,location):

    """
    Requires dataframe for specific MMSI and maybe filtered to be sufficiently close to hydrophone.

    Return lists that contain Speed values (lowest to highest),
    and a list that contains the receive level, which is calculated from the FFT of the time series data.
    Due to needing the time series data, may be time-consuming to run.
    This data can be used to get source levels by adding/making up from Tx loss using a separate function.

    Calculation for Receive Level/Sound Exposure is found in the Guerra, Thode et al. paper.

    Parameters
    ----------
    df : Pandas dataframe
        A dataframe that is filtered to only contain one MMSI. Should probably also be close to hydrophone.

    Returns
    -------
    S : a list of Speed values. When using it for Tx loss, need to make it an array and /10 to get knots.

    Rx_level : a list of receive levels
    """

    df = df.sort_values(by=['SPEED (KNOTSx10)'])
    df = df.reset_index(drop=True)

    Rx_level = []
    S = []
    file404 = 0
    for row_index, row in df.iterrows():
        try:
            inst = df['instance_id'].loc[row_index]
            #print(inst)
            st = obsread('data/'+location+'/Mseed/'+inst+'.mseed')
            data = st[-1].data
            #print(st[-1])
            
            Fs = 200
            N = len(data)
            freqs = np.linspace(-(Fs/2), Fs/2, N) # divide by 2 for nyquist, center on zero for fftshift. N points to match with N points in recording
            ft_data = np.fft.fft(data)
            idx_5hz = 63000 # index corresponding to frequency of 5 on the frequency domain centered on zero
            idx_90hz = 114000

            # Follows equation to get sound exposure
            # Guerra_Thode_et al_2011
            SE = np.sum(np.fft.fftshift(np.abs(ft_data)**2)[idx_5hz:idx_90hz] / (N * Fs))
            SEL = 10 * np.log10(SE) # dB conversion
            #print(f'SEL = {SEL}')
            Rx_level.append(SEL)
            S.append(df['SPEED (KNOTSx10)'].loc[row_index])
            #print(df['mean_distance(in kms)'].loc[row_index])
        except FileNotFoundError:
            file404 +=  1
            pass

    print(f'Files not found: {file404}')
    return S, Rx_level

# Lower and Upper Band Limits for 1/3 Octave Bands
# used for fLower and fUpper in Rx_GENERAL()
thirdOctave = {
    0: [11.2, 14.1],
    1: [14.1, 17.8],
    2: [17.8, 22.4],
    3: [22.4, 28.2],
    4: [28.2, 35.5],
    5: [35.5, 44.7],
    6: [44.7, 56.2],
    7: [56.2, 70.8],
    8: [70.8, 89.1],
}

def Rx_GENERAL(df, location, speed=False, fLower=5, fUpper=90):
    """ 
    Generalized form of Rx_level_from_df() and Rx_level_from_df_SPEED()

    Calculation for Receive Level/Sound Exposure is found in the Guerra, Thode et al. paper.

    Parameters
    ----------
    df : Pandas dataframe
        A dataframe that is filtered to only contain one MMSI. Should probably also be close to hydrophone.
    location : string
        Parameter that tells us which folder/directory, based on location, to pull our MSEED files from
    speed : boolean
        Tells us whether we want range or speed returned. True means Speed is returned while False means Range is returned
    fLower : int or float
        Tells us lower frequency bound when calculating receive level. Default is 5, just like the normal method.
    fUpper : int or float
        Tells us upper frequency bound when calculating receive level. Default is 90, just like the normal method.

    Returns
    -------
    R : a list of range values in kilometers. When using it for Tx loss, need to make it an array and *1000 to get meters.
    S : a list of Speed values. When using it for Tx loss, need to make it an array and /10 to get knots.
    Rx_level : a list of receive levels
    """
    if speed:
        #print(speed)
        df = df.sort_values(by=['SPEED (KNOTSx10)'])
        df = df.reset_index(drop=True)

        Rx_level = []
        S = []
        R = []
        file404 = 0
        for row_index, row in df.iterrows():
            try:
                inst = df['instance_id'].loc[row_index]
                #print(inst)
                st = obsread('data/'+location+'/Mseed/'+inst+'.mseed')
                data = st[-1].data
                #print(st[-1])
                
                Fs = 200
                N = len(data)
                freqs = np.linspace(-(Fs/2), Fs/2, N) # divide by 2 for nyquist, center on zero for fftshift. N points to match with N points in recording
                ft_data = np.fft.fft(data)

                abs_diff_lower = np.abs(freqs-fLower)
                abs_diff_upper = np.abs(freqs-fUpper)
                idx_lower = np.argmin(abs_diff_lower)
                idx_upper = np.argmin(abs_diff_upper)

                # Follows equation to get sound exposure
                # Guerra_Thode_et al_2011
                SE = np.sum(np.fft.fftshift(np.abs(ft_data)**2)[idx_lower:idx_upper] / (N * Fs))
                SEL = 10 * np.log10(SE) # dB conversion
                #print(f'SEL = {SEL}')
                Rx_level.append(SEL)
                S.append(df['SPEED (KNOTSx10)'].loc[row_index])
                R.append(df['distance(in km)'].loc[row_index])
                #print(df['mean_distance(in kms)'].loc[row_index])
            except FileNotFoundError:
                file404 +=  1
                pass

        print(f'Files not found: {file404}')
        return S, R, Rx_level
    else:
        df = df.sort_values(by=['distance(in km)'])
        df = df.reset_index(drop=True)

        Rx_level = []
        R = []
        file404 = 0
        for row_index, row in df.iterrows():
            try:
                inst = df['instance_id'].loc[row_index]
                #print(inst)
                st = obsread('data/'+location+'/Mseed/'+inst+'.mseed')
                data = st[-1].data
                #print(st[-1])
                
                Fs = 200
                N = len(data)
                freqs = np.linspace(-(Fs/2), Fs/2, N) # divide by 2 for nyquist, center on zero for fftshift. N points to match with N points in recording
                ft_data = np.fft.fft(data)

                abs_diff_lower = np.abs(freqs-fLower)
                abs_diff_upper = np.abs(freqs-fUpper)
                idx_lower = np.argmin(abs_diff_lower)
                idx_upper = np.argmin(abs_diff_upper)

                # Follows equation to get sound exposure
                # Guerra_Thode_et al_2011
                SE = np.sum(np.fft.fftshift(np.abs(ft_data)**2)[idx_lower:idx_upper] / (N * Fs))
                SEL = 10 * np.log10(SE) # dB conversion
                #print(f'SEL = {SEL}')
                Rx_level.append(SEL)
                R.append(df['distance(in km)'].loc[row_index])
                #print(df['mean_distance(in kms)'].loc[row_index])
            except FileNotFoundError:
                file404 +=  1
                pass

        print(f'Files not found: {file404}')
        return R, Rx_level

# Loss models from McKenna et al.
Tx_loss1 = lambda x: 20*np.log10(x*1000)

def Tx_loss2(x, location):
    depth_dict = {
            "AB": 3,
            "EC": 1.5,
            "CC": 1.5,
            "OS": 3,
            "SH": 0.8
        }
    return 20*np.log10(depth_dict[location]*1000)+10*np.log10((x*1000)/(depth_dict[location]*1000))
    
# Tx_loss3 is an adjusted method by advice of Shima after looking at the coeffs used in literature
Tx_loss3 = lambda x: 15*np.log10(x*1000)

Tx_loss1_mod = lambda x, coeff: coeff*np.log10(x*1000)
#Tx_loss2 = lambda x: 20*np.log10(3*1000)+10*np.log10((x*1000)/(3*1000))

def Tx_loss2_mod(x, location, coeff):
    depth_dict = {
            "AB": 3,
            "EC": 1.5,
            "CC": 1.5,
            "OS": 3,
            "SH": 0.8
        }
    return coeff*np.log10(depth_dict[location]*1000)+10*np.log10((x*1000)/(depth_dict[location]*1000))

def split_10m(df):
    """
    Generate start_time, end_time columns in a given dataframe so that we can run hydrophone queries.
    Start and End times are in 10 minute intervals with one second added on so that it doesn't chain together.
    When creating, didn't account for edge case where pre-existing rows would already be 10 minutes apart perfectly.
    """

    df = df.reset_index(drop=True)
    df['start_time'] = pd.to_datetime(np.zeros(len(df)))
    df['end_time'] = pd.to_datetime(np.zeros(len(df)))
    df['TIMESTAMP UTC'] = pd.to_datetime(df['TIMESTAMP UTC'])
    df.iloc[0:1]['start_time'] = df.iloc[0:1]['TIMESTAMP UTC']
    df.iloc[0:1]['end_time'] = df.iloc[0:1]['start_time'] + datetime.timedelta(minutes=10)
    for index, row in df[1:].iterrows():
        curr_time = row['TIMESTAMP UTC']
        prev_start = df.iloc[index-1]['start_time']
        prev_end = df.iloc[index-1]['end_time']
        if curr_time > prev_end:
            #row['start_time'] = curr_time
            df.at[index, 'start_time'] = curr_time
            #row['end_time'] = curr_time + datetime.timedelta(minutes=10)
            df.at[index, 'end_time'] = curr_time + datetime.timedelta(minutes=10)
        elif curr_time < prev_end:
            #row['start_time'] = prev_start
            df.at[index, 'start_time'] = prev_start
            #row['end_time'] = prev_end
            df.at[index, 'end_time'] = prev_end
    return df

def DL_Mseed_for_time_filtered_file(ais, start, end, verbose=False, node_override=None):
    """
    Inteded to work with ais dataframe with columns 'MMSI', 'SHIPNAME', 'VESSEL TYPE', 'STATUS', 'SPEED (KNOTSx10)', 'LAT',
       'LON', 'COURSE', 'HEADING', 'TIMESTAMP UTC', 'LENGTH', 'start_time',
       'end_time', 'ecdist', 'ccdist', 'abdist', 'closest hphone'

    Parameters
    ----------
    ais : Pandas dataframe
        A dataframe that aim to download the MSEEDs for
    start : string or datetime
        Start time that we filter our dataframe by
    end : string or datetime
        End time that we filter our dataframe by
    Verbose : boolean
        Boolean that determines whether we print more information as the process runs
    node_override : string
        String that tells us which hydrophone specifically to download from

    Returns
    -------
    Does not return anything. The method istead aims to query OOI and download MSEEDs into directory
    """
    
    ais = ais[(ais['TIMESTAMP UTC'] >= start) & (ais['TIMESTAMP UTC'] <= end)]

    xxdist_to_node_dict = {
        'abdist' : "AB",
        'ecdist' : "EC",
        'ccdist' : "CC",
        'AB' : 'Axial_Base',
        'EC' : 'Eastern_Caldera',
        'CC' : 'Central_Caldera'
    }

    times = {
        'start_time': ais['start_time'],
        'end_time': ais['end_time'],
        'node' : ais['closest hphone']
    }
    times = pd.DataFrame(times)
    times.drop_duplicates(subset=['start_time', 'end_time', 'node'], inplace=True)

    for index, row in times.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        str_starttime=start_time.strftime("%Y%m%d%H%M%S")
        str_endtime=end_time.strftime("%Y%m%d%H%M%S")
        node = xxdist_to_node_dict[row['node']]
        if node_override != None:
            node = node_override
        mseed_filename = 'data/AIS/Mseed/'+str(node)+'_'
        psd_pickle_filename = 'data/AIS/PSD_pickles/' + str(node)+'_'
        data_trace = ooipy.get_acoustic_data_LF(start_time, end_time, xxdist_to_node_dict[node], verbose=False, zero_mean=True)
        print(data_trace)
        if data_trace==None:
            print('data trace is none. Continuing to next')
            pass
        else:
            data_trace.write(mseed_filename+str_starttime+'_'+str_endtime+'.mseed', format='MSEED')
            if verbose:
                print('saving '+mseed_filename+str_starttime+'_'+str_endtime+'.mseed')
            try:
                psd = data_trace.compute_psd_welch(L = 256, overlap=0.3)
                psd.save(filename=psd_pickle_filename+str_starttime+'_'+str_endtime+'.json')
            except TypeError:
                print('could not compute psd for above data trace. Continuing to next')
                pass

def add_hphone_dist(ais):
    """
    Adds columns that show how close, in kilometers, an instance is from the 3 hydrophones and a column that denotes which hydrophone is the closest

    Parameters
    ----------
    ais : Pandas dataframe
        A dataframe that we aim to add columns to
    Returns
    -------
    Does not return anything. 
    """
    xxdist_to_node_dict = {
        'abdist' : "AB",
        'ecdist' : "EC",
        'ccdist' : "CC",
        'AB' : 'Axial_Base',
        'EC' : 'Eastern_Caldera',
        'CC' : 'Central_Caldera'
    }
    abdist = (45.8168, -129.7543)
    ecdist = (45.9399, -129.9742)
    ccdist = (45.9549, -130.0089)
    ais['ecdist'] = [geodesic((a, b), ecdist).miles *1.609344 for a, b in ais[['LAT', 'LON']].values]
    ais['ccdist'] = [geodesic((a, b), ccdist).miles *1.609344 for a, b in ais[['LAT', 'LON']].values]
    ais['abdist'] = [geodesic((a, b), abdist).miles *1.609344 for a, b in ais[['LAT', 'LON']].values]
    ais['closest hphone'] = ais[['ecdist', 'ccdist', 'abdist']].idxmin(axis=1).map(xxdist_to_node_dict)
    return ais

def ten_min_interp(df):
    """
    Interpolates every 10 minutes of given dataframe. Intended for use on dataframe that is filtered to one MMSI

    Parameters
    ----------
    ais : Pandas dataframe
        A dataframe that we aim to interpolate
    Returns
    -------
    Does not return anything. 
    """
    #interpolated_data = []
    new_timestamps = []

    # create intermediary timestamps at 10m1s intervals to avoid all having the same start time
    for idx, row in df[:-1].iterrows():
        #print(idx)
        curr_time = pd.to_datetime(row['TIMESTAMP UTC'])
        start_time = pd.to_datetime(row['start_time'])
        end_time = pd.to_datetime(row['end_time'])
        #next_time = pd.to_datetime(blah.iloc[idx+1]['TIMESTAMP UTC'])
        next_time = pd.to_datetime(df.at[idx+1, 'TIMESTAMP UTC'])
        #print(f'curr: {curr_time}, next: {next_time}')
        range_time = (pd.date_range(start=curr_time, end=next_time, freq='601S'))
        #print(range_time)
        new_timestamps.append(range_time)

    # interpolated times to add (includes existing timestamps)
    new_timestamps = new_timestamps[0].union_many(new_timestamps[1:])

    times_dict = {'TIMESTAMP UTC' : new_timestamps}

    # merge/join interpolated data and existing rows
    interp_df = pd.merge(df, pd.DataFrame(times_dict), on='TIMESTAMP UTC', how='outer')
    # sort all real and interpolated rows based on timestamp so we can create start/end times
    interp_df_10m = split_10m(interp_df.sort_values('TIMESTAMP UTC'))

    # interpolate LAT/LON/Speed, put back in some other data
    interp_df_10m['LAT'] = interp_df_10m['LAT'].interpolate(method='linear')
    interp_df_10m['LON'] = interp_df_10m['LON'].interpolate(method='linear')
    interp_df_10m['SPEED (KNOTSx10)'] = interp_df_10m['SPEED (KNOTSx10)'].interpolate(method='linear')
    interp_df_10m['MMSI'] = interp_df_10m.iloc[0]['MMSI']
    interp_df_10m['SHIPNAME'] = interp_df_10m.iloc[0]['SHIPNAME']
    interp_df_10m['VESSEL TYPE'] = interp_df_10m.iloc[0]['VESSEL TYPE']
    interp_df_10m['LENGTH'] = interp_df_10m.iloc[0]['LENGTH']

    # add hphone distances
    interp_df_10m = add_hphone_dist(interp_df_10m)
    return interp_df_10m