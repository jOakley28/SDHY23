""" LoadAlgorithm2.0
    Senior Design (SDHY) 2022-2023

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


def main(datafile: str):
    #put the data into a pandas dataframe
    df = pd.read_csv(datafile, sep='\t', header=None, low_memory=False)

    print('Number of rows: ', len(df))  #prints total number of rows in datafile
    
    for step_size in tqdm(STEP_SIZES):  
        #passes step with n samples through fftFilter
        if fft_sec_calc == True:  
            fft_df = pd.DataFrame()             #creates fft dataframe
            dt = (df.iloc[4,0] - df.iloc[0,0])/4    #determine dt based on change in time of the first two time steps
            ffilt = type(np.ndarray)            #create numpy non dementionalized array used to store sets before concatenation
            fftLong = []                        #empty fftLong array used to store sets during concatenation

            #create an array with the time (converted from sec to hrs) using apply in main dataframe 
            df[2] = df[0].apply(lambda x: ((x/60)/60))

            #create sets to feed fftFilter 
            for i in tqdm(range(0, len(df), step_size)):
                fft_time_set = df.iloc[i:i+step_size, 0]    #creates time array
                fft_data_set = df.iloc[i:i+step_size, 1]    #creates raw data array
        
                #runs sets through fftFilter 
                ffilt = fftFilter(dt,fft_time_set,fft_data_set,'') 

                #concatenate ffilt to fftLong
                fftLong = np.concatenate((fftLong,ffilt), axis=None)

            fft_df[0] = df[2]       #add corrected time from df[2] to fft_df[0] 
            fft_df[1] = fftLong     #put filtered data from fttLong into fft_df[1]
            fft_df[2] = df[1]       #add raw data to fft_df[2]

            #converts from ADC reading to load in kg 
            temp = []   #temporary array used to store data post conversion
            for j in range(len(fft_df.iloc[:,2])):  #converts filtered data from ADC to kg
                temp.append(linEq(fft_df.iloc[j,2].real))
                fft_df.iloc[j,2] = temp[j]

            temp = []   #temporary array used to store data post conversion
            for j in range(len(fft_df.iloc[:,1])):  #converts raw data from ADC to kg
                temp.append(linEq(fft_df.iloc[j,1].real))
                fft_df.iloc[j,1] = temp[j]

            #save ftt_df to csv
            #fft_df.to_csv(f'{saveLocation}fft_df_{step_size} for JAMES.csv', index=False, header=False)
            #testMovingAverage = moving_average(fft_df[1],step_size)
            #fftMiniPlotMTS(fft_df[0],fft_df[2],fft_df[1],fft_df[0],testMovingAverage,1,[1],[1],1,'kg',1/dt,step_size,False,'')
            
            if MTS_calc == True: #does not include MTS plots
                #creates and saves plots of raw/filtered data
                print('Plotting sectioned data without MTS overlay')
                fftMiniPlot(fft_df[0],fft_df[2],fft_df[1],'kg',1/dt,step_size,False)
            
            if MTS_calc != True: #includes MTS plots            
                print('Plotting sectioned data with MTS overlay')
                
                MTS_data, MTS_time, MTS_dt, mts_per = getMTSData(MTSfilename)
                             
                
                master_time = fft_df[0]
                #plots without error
                fftMiniPlotMTS(fft_df[0],fft_df[2],fft_df[1],MTS_time,MTS_data,MTS_dt,mts_per,0,0,0,0,'kg',1/dt,step_size,False,'E')
                
                if True == True: #resize data for error calculation 
                    MTS_data_stretch = MTS_data.copy()
                    MTS_time_stretch = MTS_time.copy()

                    resized_df = pd.DataFrame()
                    original_time = fft_df[0].copy()
                    resized_raw_df = stretchMTSData(fft_df[0],fft_df[1],MTS_time_stretch,MTS_data_stretch,step_size)
                    resized_fft_df = stretchMTSData(fft_df[0],fft_df[2],MTS_time_stretch,MTS_data_stretch,step_size)

                    error_per_raw = errorInData(resized_raw_df[2], resized_raw_df[4])
                    error_per_fft = errorInData(resized_fft_df[2], resized_fft_df[4])

                    ebtw_raw = pd.DataFrame()
                    ebtw_raw[0] = resized_raw_df[1]
                    ebtw_raw[1] = resized_raw_df[2]
                    ebtw_raw[2] = resized_raw_df[4]
                    
                    ebtw_fft = pd.DataFrame()
                    ebtw_fft[0] = resized_fft_df[1]
                    ebtw_fft[1] = resized_fft_df[2]
                    ebtw_fft[2] = resized_fft_df[4]
                    
                    if True == False:
                        plt.plot(resized_df[1],resized_df[2], label = 'raw')
                        plt.plot(resized_df[1],resized_df[4], label = 'mts')
                        #plt.plot(resized_df[1],resized_df[5], label = 'error') #use to plot diff btwn mts and raw
                        plt.fill_between(resized_df[1], resized_df[2], resized_df[4], color='orange', alpha=0.1)
                        plt.legend()
                        plt.ylim(0,110)
                        #plt.savefig(f'{FilteredAndRaw}/{filename} (TESTING MTS) - Raw load.png')
                        plt.cla()

                    #plots with error
                    fftMiniPlotMTS(original_time,fft_df[2],fft_df[1],MTS_time,MTS_data,MTS_dt,mts_per,error_per_raw,ebtw_raw,error_per_fft,ebtw_fft,'kg',1/dt,step_size,False,'')
               
    print('Sectioned Plotting Complete')

    #passes enture dataset through fftFilter
    if fft_sing_calc == True: 
        ffts_df = pd.DataFrame()            #creates ffts dataframe
        dt = (df.iloc[4,0] - df.iloc[0,0])/4    #determine dt based on change in time of the first two time steps
            
        #create an array with the time (converted from sec to hrs) using apply in main dataframe 
        df[2] = df[0].apply(lambda x: ((x/60)/60))

        #populate ffts_df with data
        ffts_df[0] = df[2]       #add corrected time from df[2] to fft_df[0] 
        ffts_df[1] = fftFilter(dt,df[0],df[1],'freq')  #runs fftFilter on entrire data set, 'freq' creates and saves plot of freq vs. power
        ffts_df[2] = df[1]       #add raw data to fft_df[2]

        #converts from ADC reading to load in kg 
        temp = []   #temporary array used to store data post conversion
        for j in range(len(ffts_df.iloc[:,2])):  #converts filtered data from ADC to kg
            temp.append(linEq(ffts_df.iloc[j,2].real))
            ffts_df.iloc[j,2] = temp[j]

        temp = []   #temporary array used to store data post conversion
        for j in range(len(ffts_df.iloc[:,1])):  #converts raw data from ADC to kg
            temp.append(linEq(ffts_df.iloc[j,1].real))
            ffts_df.iloc[j,1] = temp[j]

        #creates and saves plots of raw/filtered data
        fftMiniPlot(ffts_df[0],ffts_df[2],ffts_df[1],'kg',1/dt,'',True) 
    print('Single Plotting Complete')

#Conversion
def linEq(reading):
    #dependant on hardware 
    if circuitType == 'A':
        [m,b] = [0.03125,-105.1374375]
        [n,c] = [0.3342608,0.3342608]

    if circuitType == 'B':
        [m,b] = [1,0]   #array of [1,0] indicates a direct conversion from ADC to kg
        [n,c] = [0.007726,-77.5564]

    #converts adc reading to mV 
    mV_Reading = m*reading + b

    #converts mV to load in kg
    load = n*mV_Reading + c

    #returns load value 
    return load

#Filtering tools 
def fftFilter(dt,t,f,freqPlot):
    n = len(t)                                  #total number of samples
    fhat = np.fft.fft(f,n)                      #plot fft
    psd = fhat * np.conj(fhat) / n              #power spectrum
    freq = (1/(dt*n))*np.arange(n)              #create x-axis of frequencies
    L = np.arange(1,np.floor(n/2),dtype='int')  #only plot the first half of the frequencies

    A = np.amax((psd[L]).real)  #max power
    
    #use the psd to filter out low power freq
    c = A*0
    indicesP = psd.real > c         #find all freq with large power
    psdClean = psd * indicesP  #used to plot filtered powers 
    fhat = indicesP * fhat     #zero out all small fourier coeffs in Y    

    #filter out the high freq 
    width = 14                      #max freq (cannot exceed sample rate)
    maxFreq = np.max(freq)          #max value of freq
    fUBound = maxFreq/2 + width/2   #upper bound of freq
    fLBound = maxFreq/2 - width/2   #lower bound of freq

    indicesF = (freq > fUBound) | (freq < fLBound)  #find all freq with high freq
    psdClean = psdClean * indicesF  #used to plot filtered powers
    fhat = indicesF * fhat     #zero out all frequencies above fMax

    #apply modified fhat and take ifft
    ffilt = np.fft.ifft(fhat)  #inverse fft to get filtered signal

    #plot freq vs pwr
    if freqPlot == 'freq':
        #pltos freq vs pwr with log yaxis
        plt.title('Frequency Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (log scale)')
        plt.plot(freq[L],psd[L],color='k',linewidth=1.5,label='Raw', alpha = 0.6) 
        plt.plot(freq[L],psdClean[L],color='r',linewidth=0.5,label='Raw') 
        #plt.ylim(0,1.1*np.amax(psd[L].real))
        plt.xlim(-freq[L[-1]]/8,freq[L[-1]])
        plt.yscale('log')
        plt.ylim(0,np.amax(psd[L]))

        plt.axhline(y=c,color='k',linewidth=0.5)        #horizontal and vertical lines indicating data cutoff
        plt.axvline(x=fLBound,color='k',linewidth=0.5)
        plt.axhspan(0, c, facecolor='k', alpha=0.7)     #horizontal and vertical shading indicating data exclusion
        plt.axvspan(fLBound, freq[-1], facecolor='k', alpha=0.7)        

        plt.savefig(f'{saveLocation}/{filename} (General) - Noise Frequency (log).png')
        plt.cla()

        #plots freq vs pwr with linear yaxis
        plt.title('Frequency Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.plot(freq[L],psd[L],color='k',linewidth=1.5,label='Raw', alpha = 0.6) 
        plt.plot(freq[L],psdClean[L],color='r',linewidth=0.5,label='Raw') 
        #plt.ylim(0,1.1*np.amax(psd[L].real))
        plt.xlim(-freq[L[-1]]/8,freq[L[-1]])
        plt.ylim(0,np.amax(psd[L]/3))

        plt.axhline(y=c,color='k',linewidth=0.5)        #horizontal and vertical lines indicating data cutoff
        plt.axvline(x=fLBound,color='k',linewidth=0.5)
        plt.axhspan(0, c, facecolor='k', alpha=0.7)     #horizontal and vertical shading indicating data exclusion
        plt.axvspan(fLBound, freq[-1], facecolor='k', alpha=0.7)        

        plt.savefig(f'{saveLocation}/{filename} (General) - Noise Frequency (linear).png')
        plt.cla()

    return ffilt

def moving_average(array,step_size):
    #run with
    array = pd.DataFrame()
    array[0] = np.arange(0,100,1)
    input(moving_average(array,5))

    averaged_df = pd.DataFrame()
    output_df = pd.DataFrame()

    for i in tqdm(range(0, len(array), step_size)):
        n = len(array)
        array = array.apply(lambda x: np.real(x))

        # get the average from array
        average = array.iloc[i:i+step_size,0].mean()
        averaged_df = pd.concat([averaged_df, pd.DataFrame([[array.iloc[i,0], average]])])
    input(averaged_df)

    #map averaged_df to array
    output_df[0] = np.arange(0,n,1)
    holder = interp1d(averaged_df[0], new_len = n - step_size)
    output_df[1] = holder[:-step_size]

    return output_df

#Plot data
def fftMiniPlot(t,f,ffilt,yLim,hz,step_size,jr):
    if jr == False:
        #raw load only
        plt.plot(t,f,color='c',linewidth=0.5,label='Raw',alpha=0.7)
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)    
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Raw Data \n{testInfo}') 
        plt.title(f'@{int(hz)}hz - latency* 0s',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')
        #ylable tick marks every and lables 5
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")
        plt.savefig(f'{FilteredAndRaw}/{filename} (General) - Raw load.png')
        plt.cla() 

        #filtered load only
        plt.plot(t,ffilt,color='k',linewidth=1.0,label='Filtered')  
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Filtered Data\n{testInfo}') 
        plt.title(f'\n{step_size} samples per group\n@{int(hz)}hz - latency* ~{int(step_size/hz)}s',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")    
        plt.savefig(f'{FilteredOnly}/{filename} {step_size}a - Filtered load.png')    
        plt.cla() 

        #filtered reading ontop of row 
        plt.plot(t,f,color='c',linewidth=0.5,label='Raw',alpha=0.7)
        plt.plot(t,ffilt,color='k',linewidth=1.0,label='Filtered') 
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        #plt.xlim(t[int((len(t)-1)*0.7)],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)        
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Filtered over Raw Data\n{testInfo}') 
        plt.title(f'\n{step_size} samples per group\n@{int(hz)}hz - latency* ~{int(step_size/hz)}s',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")
        plt.savefig(f'{FilteredAndRaw}/{filename} {step_size}b - FilteredAndRaw load.png')
        plt.cla() 

    if jr == True:
        #filtered reading ontop of row 
        plt.plot(t,f,color='c',linewidth=0.5,label='Raw',alpha=0.7)
        plt.plot(t,ffilt,color='k',linewidth=1.0,label='Filtered') 
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)        
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Filtered over Raw Data\n{testInfo}') 
        plt.title(f'\n{step_size} samples per group\n@{int(hz)}hz - latency* N/A',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")
        plt.savefig(f'{saveLocation}/{filename} whole data set - FilteredAndRaw load.png')
        plt.cla()     

def fftMiniPlotMTS(t,f,ffilt,MTStime,MTSdata,MTS_dt,mts_per,error_per_r,ebtw_r,error_per_f,ebtw_f,yLim,hz,step_size,jr,E): #TODO
    if jr == False:
    #raw load only with MTS readout
        plt.plot(MTStime,MTSdata,color='r',linewidth=1.0,alpha=0.7,label=f'Applied MTS')
        if E != 'E':
            plt.plot(t,f,color='c',linewidth=0.8,label=f'Raw ({int(error_per_r*10)/10}%)')
        else: 
            plt.plot(t,f,color='c',linewidth=0.8,label='Raw')
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)    
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Raw Data \n{testInfo}') 
        plt.title(f'@{int(hz)}hz - latency* 0s',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")
        if E != 'E':
            plt.fill_between(ebtw_r[0], ebtw_r[1], ebtw_r[2], color='gray', alpha=0.2, label='error')                             #TODO 
            plt.savefig(f'{FilteredAndRaw}/{filename} (General) (with MTS and error) - Raw load.png')
        else:
            plt.title(f'Circuit {circuitType}',loc='left')
            plt.savefig(f'{FilteredAndRaw}/{filename} (General) (with MTS) - Raw load.png')
        plt.cla() 

    #filtered load only with MTS readout
        plt.plot(MTStime,MTSdata,color='r',linewidth=1.0,alpha=0.7,label=f'Applied MTS')
        if E != 'E':
            plt.plot(t,ffilt,color='k',linewidth=1.0,label=f'Filtered ({int(error_per_f*10)/10}%)') #prints filtered data with error label
        else:
            plt.plot(t,ffilt,color='k',linewidth=1.0,label='Filtered')          #prints filtered data without error label
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Filtered Data\n{testInfo}') 
        plt.title(f'\n{step_size} samples per group\n@{int(hz)}hz - latency* ~{int(step_size/hz)}s',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")  
        if E != 'E': 
            plt.fill_between(ebtw_r[0], ebtw_r[1], ebtw_r[2], color='gray', alpha=0.2, label='error')     #filles between the MTS reading and the filtred data
            plt.savefig(f'{FilteredOnly}/{filename} {step_size} (with MTS and error)a - Filtered load.png')    
        else:
            plt.savefig(f'{FilteredOnly}/{filename} {step_size} (with MTS)a - Filtered load.png')
        plt.cla() 

    #filtered reading ontop of row with MTS readout
        plt.plot(MTStime,MTSdata,color='r',linewidth=1.0,alpha=0.7,label=f'Applied MTS')
        if E != 'E':
            plt.plot(t,f,color='c',linewidth=0.8,label=f'Raw ({int(error_per_r*10)/10}%)')
            plt.plot(t,ffilt,color='k',linewidth=1.0,label=f'Filtered ({int(error_per_f*10)/10}%)') #prints filtered data with error label
        else:
            plt.plot(t,f,color='c',linewidth=0.8,label='Raw')
            plt.plot(t,ffilt,color='k',linewidth=1.0,label='Filtered')          #prints filtered data without error label
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)        
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Filtered over Raw Data\n{testInfo}') 
        plt.title(f'\n{step_size} samples per group\n@{int(hz)}hz - latency* ~{int(step_size/hz)}s',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")
        if E != 'E': 
            plt.fill_between(ebtw_r[0], ebtw_r[1], ebtw_r[2], color='gray', alpha=0.2, label='error')             
            plt.savefig(f'{FilteredAndRaw}/{filename} {step_size} (with MTS and error)b - FilteredAndRaw load.png')
        #else:
        #    plt.savefig(f'{FilteredAndRaw}/{filename} {step_size} (with MTS)b - FilteredAndRaw load.png')
        plt.cla() 

    if jr == True:
        #filtered reading ontop of row 
        plt.plot(t,f,color='c',linewidth=0.5,label='Raw')
        plt.plot(t,ffilt,color='k',linewidth=1.0,label='Filtered') 
        plt.plot(MTStime,MTSdata,color='r',linewidth=1.0,label='Applied MTS')        
        if E != 'E':
            plt.plot(master_time,errorInData,color='g',linewidth=1.0,label='Error')
        plt.minorticks_on()
        plt.grid(visible=None, which='major', linestyle='-', linewidth='1', axis='both')
        plt.grid(visible=None, which='minor', linestyle='--', linewidth='0.7', axis='y')
        plt.xlim(t[0],t[len(t)-1])
        if yLim == 'ADC':
            plt.autoscale(False, axis="y")        
            plt.ylim(6000,12000)
        if yLim == 'kg':
            plt.autoscale(False, axis="y")
            plt.ylim(0,110)        
            plt.yticks(np.arange(0,111, step=10))
        plt.legend()
        plt.title(f'Filtered over Raw Data\n{testInfo}') 
        plt.title(f'\n{step_size} samples per group\n@{int(hz)}hz - latency* N/A',loc='right',color='g')
        plt.title(f'Circuit {circuitType}',loc='left')    
        plt.xlabel("Time (hrs)")
        plt.ylabel("Load (kg)")
        if E != 'E':
            plt.savefig(f'{saveLocation}/{filename} whole data set (with MTS) - FilteredAndRaw load.png')
        else:
            plt.savefig(f'{saveLocation}/{filename} whole data set (with MTS and error) - FilteredAndRaw load.png')
        plt.cla()     

def openExplorer(loc):
    pf = pathlib.Path(f'{loc}')
    runf = f'/{pathlib.Path(*pf.parts[3:])}'
    runWf = pathlib.PureWindowsPath(runf)
    proc = subprocess.Popen(['explorer.exe', runWf])

#MTS data
def getMTSData(MTSdatafile): #TODO 
    #put the data into a pandas dataframe
    print('Reading MTS data file')
    MTS_df = pd.read_csv(MTSdatafile, sep='\t', header=None, low_memory=False)

    #print('Number of rows: ', len(MTS_df))  #prints total number of rows in datafile

    MTS_dt = (MTS_df.iloc[1,0] - MTS_df.iloc[0,0])/2    #determine dt based on change in time of the first two time steps

    #create an array with the time (converted from sec to hrs) using apply in main dataframe 
    MTS_df[2] = MTS_df[0].apply(lambda x: ((x/60)/60))

    #shift data right or left to account for inconsistant start times between devices 
    time_shift = 0.000 #time (hrs) to shift MTS data by (+:right, -:left)
    MTS_df[2] = MTS_df[2].apply(lambda x: x + time_shift) 
    MTS_time = MTS_df[2] #mts time in hrs 

    n = 10 #ratio full block area to sensor area 
    m = -1/9.81 #downward force (N) (in the negative direction) to load (kg)
    convert = m/n #conversion coefficient 
    MTS_df[3] = MTS_df[1].apply(lambda x: (x*convert)) #converts from applied mts load to local load on the sensor
    MTS_data = MTS_df[3] #mts data in kg

    return MTS_data, MTS_time, MTS_dt, n

def stretchMTSData(ARDtime,ARDdata,MTStime,MTSdata,step_size):
    holder_df = pd.DataFrame() 

    minM = np.min(MTStime)
    minA = np.min(ARDtime)
    maxM = np.max(MTStime)
    maxA = np.max(ARDtime)


    #puts -1 in all low values of the longer array to be replaced
    if minM >= minA:
        #print('MTS starts after ARD')
        for i in range(len(ARDtime)):
            if ARDtime[i] <= minM:
                ARDtime[i] = -1
            else:
                break
    else:
        #print('ARD starts after MTS') 
        for i in range(len(MTStime)):
            if MTStime[i] <= minA:
                MTStime[i] = -1
            else:
                break
    
    if maxA <= maxM:
        #print('MTS ends after ARD')
        for i in range(len(MTStime)):
            if MTStime[i] >= maxA:
                MTStime[i] = -1
    else:
        #print('ARD ends after MTS')
        for i in range(len(ARDtime)):
            if ARDtime[i] >= maxM:
                ARDtime[i] = -1

    #removes all -1s
    ARDtime_clean = []
    MTStime_clean = []
    ARDdata_clean = []
    MTSdata_clean = []

    for i in range(len(ARDtime)):
        if ARDtime[i] != -1:
            ARDtime_clean.append(ARDtime[i])
            ARDdata_clean.append(ARDdata[i])
    for i in range(len(MTStime)):
        if MTStime[i] != -1:
            MTStime_clean.append(MTStime[i])
            MTSdata_clean.append(MTSdata[i])


    #find the smaller length
    new_len = min(len(ARDtime_clean),len(MTStime_clean))

    ARDresize_df = pd.DataFrame() 
    MTSresize_df = pd.DataFrame() 

    ARDresize_df[0] = ARDtime_clean 
    ARDresize_df[1] = ARDdata_clean 
    MTSresize_df[0] = MTStime_clean 
    MTSresize_df[1] = MTSdata_clean 

    n = len(ARDresize_df)
    m = len(MTSresize_df)

    #new_len = int(len(toResize_df)*1.7) 
    holder_df[0] = np.arange(0,new_len,1)
    
    #np.interp(np.linspace(0, n - 1, num=new_len), np.arange(n), toResize_df)
    holder_df[1] = interp1d(ARDresize_df[0], new_len)
    holder_df[2] = interp1d(ARDresize_df[1], new_len)
    holder_df[3] = interp1d(MTSresize_df[0], new_len)
    holder_df[4] = interp1d(MTSresize_df[1], new_len)

    return holder_df

def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

#Analytical tools
def errorInData(a, b):
    c = []
    error = []    
    for i in range(len(a)):
        c.append(abs(a[i] - b[i]))
        error.append(100*c[i]/b[i])
    avgError = int(abs(100*np.mean(error)))/100
    return avgError

if __name__ == "__main__":  

    if True == True:
        #argument parser
        parser = ArgumentParser()
        parser.add_argument("-o", "--open",dest="open", help="1: opens explorer window only, 2: open explor after run ", default=0,type=int)
        parser.add_argument("-s", "--stepSizes", dest="stepSizes", nargs='+', help="step sizes",default=[100],type=int)
        parser.add_argument("-c", "--circuitType", dest="circuitType", help="circuit type", default="A")
        parser.add_argument("-l", "--letter", dest="letter", help="test letter", default="DEF a")
        parser.add_argument("-d", "--description", dest="description",help="short test description", default="- Static Load")
        parser.add_argument("-i", "--testInfo", dest="testInfo", help="longer test description", default="DEF: Static loading with user induced noise")
        parser.add_argument("-D", "--datafile", dest="datafile", help="datafile path",default='')
        parser.add_argument("-S", "--saveLocation", dest="saveLocation", help="saveLocation path",default='')
        parser.add_argument("-y", "--fft_section_bool", dest="fft_section_bool", help="enables running sectioned sample analysis", default=True)
        parser.add_argument("-z", "--fft_single_bool", dest="fft_single_bool", help="enables running single sample analysis", default=True)
        parser.add_argument("-m", "--mts_bool", dest="mts_bool", help="enables plotting MTS data with fft data", default=True) #TODO may need to invert bool value
        parser.add_argument("-M", "--mts_datafile", dest="mts_datafile", help="MTS datafile path",default='')


        args = parser.parse_args()

        STEP_SIZES = (args.stepSizes) #stepsizes to compute

        circuitType = args.circuitType    #based on hardware setup
        test_ltr = args.letter          #test letter
        letter = f'Test {test_ltr}'     #test letter for file names

        description = (str(args.description)[1:-1]).replace('\'','') #short description of test (for file names)
        testInfo = (str(args.testInfo)[1:-1]).replace(',','') #additional information (for charts)

        datafile = pathlib.Path(args.datafile) #location to pull datafile from
        test_key = f'{letter} - {date.today()} {description} (Circuit {circuitType})' 
        saveLocation = f'{pathlib.PurePosixPath(args.saveLocation)}/{test_key}/' #location to save datafile to
        MTSfilename = pathlib.Path(args.mts_datafile) #location to pull MTS datafile from 

        fft_sec_calc = args.fft_section_bool; fft_sing_calc = args.fft_single_bool;   #determines which filters are run
        
        MTS_calc = args.mts_bool #determines if MTS data is plotted

        filename = test_key

        #open explorer if requested
        if args.open == 1:
            pf = pathlib.Path(f'{saveLocation}')
            runf = f'/{pathlib.Path(*pf.parts[3:-1])}'
            runWf = pathlib.PureWindowsPath(runf)
            subprocess.Popen(['explorer.exe', runWf])    
            exit(1)


        if not os.path.isfile(datafile):
            print("Data file not found")
            exit(1)

        if MTS_calc != True:
            if not os.path.isfile(MTSfilename):
                print("MTS file not found")
                exit(1)            
        

        #if not os.path.isdir(saveLocation):
            #print("Save directory not found")
            #exit(1)
        
        FilteredAndRaw = f'{saveLocation}Filtered and Raw/'
        FilteredOnly = f'{saveLocation}Filtered Only/'
        name = f'{test_key}  Filtered load {STEP_SIZES[0]}.png'
        fullString = f'{saveLocation}{name}'

    #create folders as nessesary
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    if not os.path.exists(FilteredAndRaw):
        os.makedirs(FilteredAndRaw)
    if not os.path.exists(FilteredOnly):
        os.makedirs(FilteredOnly)
    
    #get last element from datafile path
    filename = datafile.parts[-1]


    #copy datafile to test directory
    try:
        shutil.copy(datafile, saveLocation)
    except shutil.SameFileError:
        print("Data file already exists in this directory")

    #rename datafile to test directory
    if MTS_calc != True:
        try:
            shutil.copy(MTSfilename, saveLocation)
        except shutil.SameFileError:
            print("MTS file already exists in this directory")

    #rename datafile to 'DATALOG.txt'
    
    finalDataFileName = f'{saveLocation}{test_key} SensorData.txt'
    os.rename(f'{saveLocation}/{datafile.parts[-1]}', f'{finalDataFileName}')
    newFileName = f'{saveLocation}{test_key} DATALOG.txt'

    #rename MTS datafile to 'MTS.txt'
    if MTS_calc != True:
        os.rename(f'{saveLocation}/{MTSfilename.parts[-1]}', f'{saveLocation}{test_key} MTS.txt')
        newMTSName = f'{saveLocation}{test_key} MTS.txt'

    #create new text file with the arg info
    with open(f'{saveLocation}info.txt', 'w') as f:
        f.write(f'created on {date.today()}\n\n')
        f.write(f'-s, --stepSizes: {args.stepSizes}\n')
        f.write(f'-c, --circuitType: {args.circuitType}\n')
        f.write(f'-l, --letter: {args.letter}\n')
        f.write(f'-d, --description: {args.description}\n')
        f.write(f'-i, --testInfo: {args.testInfo}\n')
        f.write(f'-D, --datafile: {finalDataFileName}\n')
        f.write(f'-S, --saveLocation: {args.saveLocation}\n')
        f.write(f'-y, --fft_section_bool: {args.fft_section_bool}\n')
        f.write(f'-z, --fft_single_bool: {args.fft_single_bool}\n')
        f.write(f'-m, --mts_bool: {args.mts_bool}\n')
        f.write(f'-M, --mts_datafile: {args.mts_datafile}\n\n\n')

        f.write(f'Copy and paste the following command to run this test again:\n\n')
        stepHolder = (str(args.stepSizes)[1:-1]).replace(',','')
        if MTS_calc == True:
            f.write(f'python3 LoadAlgorithm1.16.py -c {args.circuitType} -l {args.letter} -d \'{args.description}\' -i \'{args.testInfo}\' -D \'{finalDataFileName}\' -S \'{args.saveLocation}\' -s {stepHolder}\n')
        else:
            f.write(f'python3 LoadAlgorithm1.16.py -c {args.circuitType} -l {args.letter} -d \'{args.description}\' -i \'{args.testInfo}\' -D \'{finalDataFileName}\' -S \'{args.saveLocation}\' -m 1 -M {newMTSName} -s {stepHolder}\n')


    # python3 LoadAlgorithm1.16.py -c B -l d -d ' - unknown ' -i ' unknown demo  ' -D '/mnt/c/Users/jacob/OneDrive/Documents/George Fox 2019-2023/2023a Spring/Senior Design 2/Datalogs/1-22-23 near static noise.txt' -S '/mnt/c/Users/jacob/OneDrive/Documents/George Fox 2019-2023/2023a Spring/Senior Design 2/Datalogs/Monday 1-23-23/' -s 200 -m 1 -M '/mnt/c/Users/jacob/OneDrive/Documents/George Fox 2019-2023/2023a Spring/Senior Design 2/Datalogs/MTS Test 1.txt' -o 2

    main(datafile)
    if args.open == 2:
        openExplorer(saveLocation)
