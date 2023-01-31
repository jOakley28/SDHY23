""" LoadAlgorithm2.0
    
    Senior Design (SDHY) 2022-2023
    George Fox University 

    Jacob Hankland
    jhankland19@georgefox.edu
"""

#import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import subprocess 
from tqdm import tqdm
from argparse import ArgumentParser #TODO
from datetime import date
import pathlib
import warnings

#SETUP
plt.rcParams['figure.figsize'] = [20, 12]   #sets figure size
plt.rcParams.update({'font.size': 18})      #sets font size
warnings.filterwarnings('ignore')           #hide warnings


def main():
    pass


#Import data
def import_arduino_data(arduinoDataFile,ardSepBy):  #get arduino data from CSV file
    #put the data into a pandas dataframe
    arduino_data_df = pd.read_csv(arduinoDataFile, sep=ardSepBy, header=1, low_memory=False)
    print('Number of rows: ', len(arduino_data_df[0]))  #prints total number of rows in datafile

    arduino_data_df[0] = arduino_data_df[0].apply(lambda x: x/3600)   #converts column 0 from seconds to hours

    arduino_data_dt = (arduino_data_df.iloc[4,0] - arduino_data_df.iloc[0,0])/4    #determine dt based on change in time of the first four time steps
    arduino_data_freq = int(1/arduino_data_dt)

    #for testing
    print(f'Arduino sampeling frequency {arduino_data_freq}hz')   #prints the sampling frequency in hz
    input(arduino_data_df[:2]) #prints first 2 rows of arduino data

    return arduino_data_df[2:], arduino_data_freq #time (hours) & ADC (kg), arduino sample frequency

def import_mts_data(mtsDataFile,mtsSepBy,mtsTimeShift,areaRatio):  #get mts data from CSV file
    mts_data_df = pd.read_csv(mtsDataFile, sep=mtsSepBy, header=None, low_memory=False)

    mts_data_dt = (mts_data_df.iloc[4,0] - mts_data_df.iloc[0,0])/4    #determine dt based on change in time of the first four time steps
    mts_data_freq = int(1/mts_data_df)

    #correct time
    MTS_df[0] = MTS_df[0].apply(lambda x: x + mtsTimeShift)   #shift data right or left to account for inconsistant start times between devices - SECONDS!!
    MTS_df[0] = MTS_df[0].apply(lambda x: ((x/60)/60))  #converts column 0 from seconds to hours

    nToKg = -1/9.806 #downward force (N) (in the negative direction) to load (kg)
    convert = nToKg/areaRatio #conversion coefficient 
    MTS_df[1] = MTS_df[1].apply(lambda x: (x*convert)) #converts from applied mts load to local load on the sensor

    #for testing
    print(f'MTS sampeling frequency: {mts_data_freq}hz')
    input(mts_df[:2]) #prints first 2 row of mts data

    return MTS_df[1:], mts_data_freq #time (hours) & mts applied laoding (kg), mts sample frequency


#Determine if there was a change in loading 
def detect_change_in_load():    #determine if there was a change in loading 
    pass

def take_derivative():  #run again for 2nd derivative 
    pass

def write_new_load():   #based on detect_chagne_in_load, write new loading 
    pass

def convert_adc_to_kg():    #convert adc to kg (using change in ADC)
    pass


#Error approximation 
def errorInData(sampleA, sampleB):  #determine the error (both instantenous and average)
    c = []
    errorC = []    
    for i in range(len(sampleA)):
        c.append(abs(sampleA[i] - sampleB[i]))
        errorC.append(100*c[i]/sampleB[i])
    avgError = int(abs(100*np.mean(error)))/100
    return avgError, errorC  #average error, error at each data point 

def stretch_data(sampleAx,sampleAy,sampleBx,sampleBy):    #makes both arrays the same length (used for error calculation)
    holder_df = pd.DataFrame() 

    #determine the size of the two samples
    minM = np.min(sampleBx)
    minA = np.min(sampleAx)
    maxM = np.max(sampleBx)
    maxA = np.max(sampleAx)

    #puts -1 in all low values of the longer array to be replaced
    if minM >= minA:
        #print('MTS starts after ARD')
        for i in range(len(sampleAx)):
            if sampleAx[i] <= minM:
                sampleAx[i] = -1
            else:
                break
    else:
        #print('ARD starts after MTS') 
        for i in range(len(sampleBx)):
            if sampleBx[i] <= minA:
                sampleBx[i] = -1
            else:
                break
    
    if maxA <= maxM:
        #print('MTS ends after ARD')
        for i in range(len(sampleBx)):
            if sampleBx[i] >= maxA:
                sampleBx[i] = -1
    else:
        #print('ARD ends after MTS')
        for i in range(len(sampleAx)):
            if sampleAx[i] >= maxM:
                sampleAx[i] = -1

    #removes all -1s
    sampleAx_clean = []
    sampleBx_clean = []
    sampleAy_clean = []
    sampleBy_clean = []

    for i in range(len(sampleAx)):
        if sampleAx[i] != -1:
            sampleAx_clean.append(sampleAx[i])
            sampleAy_clean.append(sampleAy[i])
    for i in range(len(sampleBx)):
        if sampleBx[i] != -1:
            sampleBx_clean.append(sampleBx[i])
            sampleBy_clean.append(sampleBy[i])


    #find the smaller sample
    new_len = min(len(sampleAx_clean),len(sampleBx_clean))

    sampleAresize_df = pd.DataFrame() 
    sampleBresize_df = pd.DataFrame() 

    sampleAresize_df[0] = sampleAx_clean 
    sampleAresize_df[1] = sampleAy_clean 
    sampleBresize_df[0] = sampleBx_clean 
    sampleBresize_df[1] = sampleBy_clean 

    #new_len = int(len(toResize_df)*1.7) 
    holder_df[0] = np.arange(0,new_len,1)
    
    #np.interp(np.linspace(0, n - 1, num=new_len), np.arange(n), toResize_df)
    holder_df[1] = interp1d(sampleAresize_df[0], new_len)
    holder_df[2] = interp1d(sampleAresize_df[1], new_len)
    holder_df[3] = interp1d(sampleBresize_df[0], new_len)
    holder_df[4] = interp1d(sampleBresize_df[1], new_len)

    return holder_df[1:]    #Ax, Ay, Bx, By

def interpolate(array: np.ndarray, new_len: int) -> np.ndarray: #used in stretch data
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


#Output
def plot_r():   #plot only the raw data
    pass

def plot_r_and_f(): #plot both raw and filtered data
    pass

def open_output_in_folder(loc):    #opens the folder where data was saved 
    pf = pathlib.Path(f'{loc}')
    runLf = f'/{pathlib.Path(*pf.parts[3:])}'   #trims address to only include what is relevent for windows address 
    runWf = pathlib.PureWindowsPath(runLf)      #converts to windows path 
    proc = subprocess.Popen(['explorer.exe', runWf])


if __name__ == "__main__":  
    main()