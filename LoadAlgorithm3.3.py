""" LoadAlgorithm2.0
    
    Senior Design (SDHY) 2022-2023
    George Fox University 

    Jacob Hankland
    jhankland19@georgefox.edu
"""


""" - - - - - -Order of Opperations- - - - - -
    *Parse command line argument
    *Configure save directory 

    *Main Process
        Get arduino (ADC) data
        Detect change in ADC reading
            derv 1
            derv 2
        Convert ADC reading to kg units

        Get MTS data
        Correct MTS data 
        
        Calculate Error
            Stretch datasets to be the same length
            Take the diffrence between the two curves 
        
        Get plot/file names
        Plot the original ADC vs MTS load (also include error vs time)
        Plot corrected ADC vs MTS load (also include error vs time)
        Plot both ADC readings vs MTS load

    *Save info file
"""

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import subprocess
from tqdm import tqdm
from argparse import ArgumentParser  # TODO
from datetime import date
import pathlib
import warnings
from gooey import Gooey, GooeyParser


# SETUP
plt.rcParams["figure.figsize"] = [20, 12]  # sets figure size
plt.rcParams.update({"font.size": 18})  # sets font size
# warnings.filterwarnings("ignore")  # hide warnings


# main process
def main_process():
    # import data 
    rawAdcData = pd.DataFrame()
    mtsData = pd.DataFrame()
    
    rawAdcData, rawAdcFreq = import_arduino_data(
        arduinoDataFile, ardSepBy
    )  # import arduino data
    mtsData, mtsFreq = import_mts_data(
        mtsDataFile, mtsSepBy, mtsTimeShift
    )  # imports mts data
    
    # detect change in load based on ADC data
    correctedAdcData = detect_change_in_load(rawAdcData)  # TODO if broken replace with detect_change_in_load(rawAdcDataLoadCellError) - looks for change in load

    # convert values to kg
    rawAdcData[1] = convert_adc_to_kg(rawAdcData[1]) # converts ADC to kg
    correctedAdcData[1] = convert_adc_to_kg(correctedAdcData[1]) # converts ADC to kg
    mtsData[1] = correct_mts_data(mtsData[1])  # corrects mts (-N to +kg) (includes area ratio)


    # load cell frame plots 
    if loadCellFramePlot == True:
        print("Plot load in the load cell frame:")
        # create copies for error calculation 
        rawAdcDataLoadCellError = rawAdcData.copy() 
        correctedAdcDataLoadCellError = correctedAdcData.copy()  
        mtsDataLoadCellError = mtsData.copy() 

        # convert ADC value to kg
        rawAdcDataLoadCellError[1] = convert_adc_to_kg(rawAdcDataLoadCellError[1])
        correctedAdcDataLoadCellError[1] = convert_adc_to_kg(correctedAdcDataLoadCellError[1])

        # calculate percent error
        print("Estimating error...")
        rAdcMtsError, rAvgError, rEbtw = errorCalculation(
            rawAdcDataLoadCellError, mtsDataLoadCellError
        )  # raw ADC vs MTS precent error
        cAdcMtsError, cAvgError, cEbtw = errorCalculation(
            correctedAdcDataLoadCellError, mtsDataLoadCellError
        )  # corrected ADC vs MTS precent error


        # create copies to convert to load cell frame
        rawAdcDataLoadCell = rawAdcData.copy() 
        correctedAdcDataLoadCell = correctedAdcData.copy()  
        mtsDataLoadCell = mtsData.copy() 

        # plot data 
        print("Plotting data...")
        rawMts_str, correctedMts_str, rawcorrectedMts_str = new_file_names('load_cell')
        ymax = 110
        plot_two(
            rawAdcDataLoadCell,
            mtsDataLoadCell,
            rawAdcFreq,
            rAvgError,
            rAdcMtsError,
            rEbtw,
            rawMts_str,
            ymax, 
        )  # plot raw ADC and MTS

        if dervThresh != False:  # from argprse - only runs if using FlexiForce sensor
            plot_two(
                correctedAdcDataLoadCell,
                mtsDataLoadCell,
                rawAdcFreq,
                cAvgError,
                cAdcMtsError,
                cEbtw,
                correctedMts_str,
                ymax,
            )  # plot corrected ADC and MTS

            plot_three(
                rawAdcDataLoadCell,
                correctedAdcDataLoadCell,
                mtsDataLoadCell,
                rawAdcFreq,
                rAvgError,
                cAvgError,
                rawcorrectedMts_str,
                ymax,
            )  # plot raw and corrected ADC along with MTS

        print("Plotting complete\n")
    
    # block frame plots 
    if blockFramePlot == True:
        print("Plot load in the block frame:")
        # create copies for error calculation 
        rawAdcDataBlockError = rawAdcData.copy() 
        correctedAdcDataBlockError = correctedAdcData.copy()  
        mtsDataBlockError = mtsData.copy() 

        # convert ADC value to kg
        rawAdcDataBlockError[1] = convert_adc_to_kg(rawAdcDataBlockError[1])
        correctedAdcDataBlockError[1] = convert_adc_to_kg(correctedAdcDataBlockError[1])

        # convert error to block frame
        rawAdcDataBlockError[1], correctedAdcDataBlockError[1], mtsDataBlockError[1] = laod_cell_to_block_frame(rawAdcDataBlockError[1], correctedAdcDataBlockError[1], mtsDataBlockError[1])

        # calculate percent error
        print("Estimating error...")
        rAdcMtsError, rAvgError, rEbtw = errorCalculation(
            rawAdcDataBlockError, mtsDataBlockError
        )  # raw ADC vs MTS precent error
        cAdcMtsError, cAvgError, cEbtw = errorCalculation(
            correctedAdcDataBlockError, mtsDataBlockError
        )  # corrected ADC vs MTS precent error


        # create copies to convert to block frame
        rawAdcDataBlock = rawAdcData.copy() 
        correctedAdcDataBlock = correctedAdcData.copy()  
        mtsDataBlock = mtsData.copy() 

        # convert from load cell frame 
        rawAdcDataBlock[1], correctedAdcDataBlock[1], mtsDataBlock[1] = laod_cell_to_block_frame(rawAdcDataBlock[1], correctedAdcDataBlock[1], mtsDataBlock[1])

        # plot data 
        print("Plotting data...")
        rawMts_str, correctedMts_str, rawcorrectedMts_str = new_file_names('block')
        ymax = 1000
        plot_two(
            rawAdcDataBlock,
            mtsDataBlock,
            rawAdcFreq,
            rAvgError,
            rAdcMtsError,
            rEbtw,
            rawMts_str,
            ymax, 
        )  # plot raw ADC and MTS

        if dervThresh != False:  # from argprse - only runs if using FlexiForce sensor
            plot_two(
                correctedAdcDataBlock,
                mtsDataBlock,
                rawAdcFreq,
                cAvgError,
                cAdcMtsError,
                cEbtw,
                correctedMts_str,
                ymax,
            )  # plot corrected ADC and MTS

            plot_three(
                rawAdcDataBlock,
                correctedAdcDataBlock,
                mtsDataBlock,
                rawAdcFreq,
                rAvgError,
                cAvgError,
                rawcorrectedMts_str,
                ymax,
            )  # plot raw and corrected ADC along with MTS

        print("Plotting complete\n")

    # rearAxle frame plots 
    if rearAxleFramePlot == True:
        print("Plot load in the rear axle frame:")
        # create copies for error calculation 
        rawAdcDataRearAxleError = rawAdcData.copy() 
        correctedAdcDataRearAxleError = correctedAdcData.copy()  
        mtsDataRearAxleError = mtsData.copy() 

        # convert ADC value to kg
        rawAdcDataRearAxleError[1] = convert_adc_to_kg(rawAdcDataRearAxleError[1])
        correctedAdcDataRearAxleError[1] = convert_adc_to_kg(correctedAdcDataRearAxleError[1])

        # convert error to axle frame
        rawAdcDataRearAxleError[1], correctedAdcDataRearAxleError[1], mtsDataRearAxleError[1] = laod_cell_to_rearAxle_frame(rawAdcDataRearAxleError[1], correctedAdcDataRearAxleError[1], mtsDataRearAxleError[1])

        # calculate percent error
        print("Estimating error...")
        rAdcMtsError, rAvgError, rEbtw = errorCalculation(
            rawAdcDataRearAxleError, mtsDataRearAxleError
        )  # raw ADC vs MTS precent error
        cAdcMtsError, cAvgError, cEbtw = errorCalculation(
            correctedAdcDataRearAxleError, mtsDataRearAxleError
        )  # corrected ADC vs MTS precent error


        # create copies to convert to rearAxle  frame
        rawAdcDataRearAxle = rawAdcData.copy() 
        correctedAdcDataRearAxle = correctedAdcData.copy()  
        mtsDataRearAxle = mtsData.copy() 

        # convert from load cell frame 
        rawAdcDataRearAxle[1], correctedAdcDataRearAxle[1], mtsDataRearAxle[1] = laod_cell_to_rearAxle_frame(rawAdcDataRearAxle[1], correctedAdcDataRearAxle[1], mtsDataRearAxle[1])

        # plot data 
        print("Plotting data...")
        rawMts_str, correctedMts_str, rawcorrectedMts_str = new_file_names('rearAxle')
        ymax = 3500
        plot_two(
            rawAdcDataRearAxle,
            mtsDataRearAxle,
            rawAdcFreq,
            rAvgError,
            rAdcMtsError,
            rEbtw,
            rawMts_str,
            ymax, 
        )  # plot raw ADC and MTS

        if dervThresh != False:  # from argprse - only runs if using FlexiForce sensor
            plot_two(
                correctedAdcDataRearAxle,
                mtsDataRearAxle,
                rawAdcFreq,
                cAvgError,
                cAdcMtsError,
                cEbtw,
                correctedMts_str,
                ymax,
            )  # plot corrected ADC and MTS

            plot_three(
                rawAdcDataRearAxle,
                correctedAdcDataRearAxle,
                mtsDataRearAxle,
                rawAdcFreq,
                rAvgError,
                cAvgError,
                rawcorrectedMts_str,
                ymax,
            )  # plot raw and corrected ADC along with MTS

        print("Plotting complete\n")


# Convert between frames
def laod_cell_to_block_frame(raw_arduino_load_cell_frame,corrected_arduino_load_cell_frame,mts_load_cell_frame):
    # aera ratio 
    blockArea = blockLength * blockWidth  # difined by user (in^2)
    areaRatio = blockArea / sensorArea  # from data sheet (in^2)

    raw_arduino_to_block = pd.DataFrame()
    raw_arduino_to_block[1] = raw_arduino_load_cell_frame
    raw_arduino_block_frame = raw_arduino_to_block[1].apply(lambda x: x*areaRatio)

    corrected_arduino_to_block = pd.DataFrame()
    corrected_arduino_to_block[1] = corrected_arduino_load_cell_frame
    corrected_arduino_block_frame = corrected_arduino_to_block[1].apply(lambda x: x*areaRatio)

    mts_to_block = pd.DataFrame()
    mts_to_block[1] = mts_load_cell_frame
    mts_block_frame = mts_to_block[1].apply(lambda x: x*areaRatio)
    return raw_arduino_block_frame, corrected_arduino_block_frame, mts_block_frame


def laod_cell_to_rearAxle_frame(raw_arduino_load_cell_frame,corrected_arduino_load_cell_frame,mts_load_cell_frame):
    # aera ratio 
    blockArea = blockLength * blockWidth  # difined by user (in^2)
    areaRatio = blockArea / sensorArea  # from data sheet (in^2)
    block_ratio = 1/0.241 # from FBD
    
    raw_arduino_to_rearAxle = pd.DataFrame()
    raw_arduino_to_rearAxle[1] = raw_arduino_load_cell_frame
    raw_arduino_rearAxle_frame = raw_arduino_to_rearAxle[1].apply(lambda x: x*areaRatio*block_ratio)

    corrected_arduino_to_rearAxle = pd.DataFrame()
    corrected_arduino_to_rearAxle[1] = corrected_arduino_load_cell_frame
    corrected_arduino_rearAxle_frame = corrected_arduino_to_rearAxle[1].apply(lambda x: x*areaRatio*block_ratio)

    mts_to_rearAxle = pd.DataFrame()
    mts_to_rearAxle[1] = mts_load_cell_frame
    mts_rearAxle_frame = mts_to_rearAxle[1].apply(lambda x: x*areaRatio*block_ratio)
    return raw_arduino_rearAxle_frame, corrected_arduino_rearAxle_frame, mts_rearAxle_frame


# Import data
def import_arduino_data(arduinoDataFile, ardSepBy):  # get arduino data from CSV file
    print("Getting arduino data...")
    # put the data into a pandas dataframe
    arduino_data_df = pd.read_csv(
        arduinoDataFile, sep=ardSepBy, header=None, low_memory=False
    )

    arduino_data_df[1] = arduino_data_df[1].apply(
        lambda x: x / scale
    )  # scales by factor defined in argparse

    arduino_data_df[0] = arduino_data_df[0].apply(
        lambda x: x / 3600
    )  # converts column 0 from seconds to hours

    arduino_data_dt = (
        arduino_data_df.iloc[4, 0] - arduino_data_df.iloc[0, 0]
    ) / 5  # determine dt based on change in time of the first four time steps
    arduino_data_freq = int(1 / (3600 * arduino_data_dt))

    print(
        f"{len(arduino_data_df[0])} rows\n"
    )  # prints total number of rows in datafile
    return (
        arduino_data_df,
        arduino_data_freq,
    )  # time (hours) & ADC (kg), arduino sample frequency


def import_mts_data(mtsDataFile, mtsSepBy, mtsTimeShift):  # get mts data from CSV file
    print("Gettting mts data...")
    # put the data into a pandas dataframe
    mts_data_df = pd.read_csv(mtsDataFile, sep=mtsSepBy, header=None, low_memory=False)

    mts_data_dt = (
        mts_data_df.iloc[4, 0] - mts_data_df.iloc[0, 0]
    ) / 5  # determine dt based on change in time of the first four time steps
    mts_data_freq = int(1 / mts_data_dt)

    # correct time
    mts_data_df[0] = mts_data_df[0].apply(
        lambda x: x + mtsTimeShift
    )  # shift data right or left to account for inconsistant start times between devices - SECONDS!!
    mts_data_df[0] = mts_data_df[0].apply(
        lambda x: ((x / 60) / 60)
    )  # converts column 0 from seconds to hours

    print(f"{len(mts_data_df[0])} rows\n")  # prints total number of rows in datafile

    return (
        mts_data_df[0:],
        mts_data_freq,
    )  # time (hours) & mts applied laoding (kg), mts sample frequency


def correct_mts_data(rawMtsData):
    blockArea = blockLength * blockWidth  # difined by user (in^2)
    # sensorArea = 0.785852  # from data sheet (in^2)
    areaRatio = blockArea / sensorArea
    nToKg = -1 / 9.806  # downward force (N) (in the negative direction) to load (kg)
    corFactor = nToKg / (areaRatio) 
    correctedMtsData = rawMtsData.apply(lambda x: x * corFactor)
    return correctedMtsData


# Determine if there was a change in loading
def detect_change_in_load(data):  # determine if there was a change in loading
    dataToDetect = data.copy()
    detectedAdc = pd.DataFrame()
    detectedAdc[0] = data[0]
    adc_corrected = []

    if dervThresh != False:
        step_size = 30
        for i in range(len(data[1])):
            if i > step_size:
                time_group_step_size = data[0][i - step_size : i].tolist()
                data_group_step_size = data[1][i - step_size : i].tolist()
                first_derv = take_first_derivative(
                    time_group_step_size, data_group_step_size
                )

                if (
                    first_derv.mean() < dervThresh and first_derv.mean() > 0
                ):  # if under the derv threshold, use the prev value
                    adc_corrected.append(adc_corrected[i - 1])
                else:  # if above the derv threshold, use the current value
                    adc_corrected.append(data[1][i])
            else:
                adc_corrected.append(data[1][i])  # add the first value to the list
        detectedAdc[1] = adc_corrected

    else:
        detectedAdc[1] = dataToDetect[1]

    return detectedAdc

    """if 2 in data[1]:
        print('Arduino datafile contains 2nd data column')
        for i in range(len(data[1])):
            adc_corrected.append(int(data[1][i]))
        detectedAdc[1] = adc_corrected
    else:
        print('Arduino datafile only contains 1 data column')
        dataToDetect[3] = dataToDetect[2]
        detectedAdc[0] = data[1]
    """


def take_first_derivative(time, data):  # takes the first derivative of a dataset
    dataFirstDir = []  # creates array ofset by 1 to account for 1st dir offset
    dataDerv_df = pd.DataFrame()
    dataDerv_df[0] = []
    for i in range(
        len(data)
    ):  # takes the difrence between each two data points and divides by the change in time
        if i > 0:
            ithDerivative = (data[i] - data[i - 1]) / (time[i] - time[i - 1])
            dataFirstDir.append(
                ithDerivative / 3600
            )  # appends the derivative array with the next ith derivative
    dataDerv_df = pd.DataFrame()
    dataDerv_df[0] = dataFirstDir
    return dataDerv_df[0]


def take_second_derivative(
    time, dataFirstDir
):  # takes the second derivative of a dataset
    dataSecondDir = [0, 0]  # creates array ofset by 2 to account for 2nd dir offset
    for i in range(
        len(dataFirstDir) - 1
    ):  # takes the difrence between each two data points and divides by the change in time
        ithDerivative = (dataFirstDir[i + 1] - dataFirstDir[i]) / (
            time[i + 1] - time[i]
        )
        dataSecondDir = dataSecondDir.append(
            ithDerivative
        )  # appends the derivative array with the next ith derivative
    return dataSecondDir


def write_new_load():  # based on detect_chagne_in_load, write new loading
    pass


def convert_adc_to_kg(adcData):  # TODO convert adc to kg (using change in ADC)
    # [m, b] = [0.007726, -77.5564]  # circut B profile
    [m, b] = [adcSlope, adcOffset]
    adcDataKg = adcData.apply(lambda x: m * x + b)
    return adcDataKg


# Error approximation
def errorCalculation(sampleA, sampleB):
    resized_df = stretch_data(sampleA[0], sampleA[1], sampleB[0], sampleB[1])
    ebtw = pd.DataFrame()  # bounds for fill between in plots
    ebtw[0] = resized_df[0]
    ebtw[1] = resized_df[1]
    ebtw[2] = resized_df[3]

    errorC, avgError = errorBetweenCurves(resized_df[1], resized_df[3])
    return errorC, avgError, ebtw


def errorBetweenCurves(
    sampleA, sampleB
):  # determine the error (both instantenous and average)\
    c = []
    errorC = []
    for i in range(len(sampleA)):
        c.append(abs(sampleA[i] - sampleB[i]))
        errorC.append(100 * c[i] / sampleB[i])
    avgError = int(abs(100 * np.mean(errorC))) / 100
    return errorC, avgError  # average error, error at each data point


def stretch_data(
    sampleAx: np.array, sampleAy: np.array, sampleBx: np.array, sampleBy: np.array
):  # makes both arrays the same length (used for error calculation)
    # determine the size of the two samples
    minM = np.min(sampleBx)
    minA = np.min(sampleAx)
    maxM = np.max(sampleBx)
    maxA = np.max(sampleAx)

    # puts -1 in all low values of the longer array to be replaced
    if minM >= minA:
        # print('MTS starts after ARD')
        for i in range(len(sampleAx)):
            if sampleAx[i] <= minM:
                sampleAx[i] = -1
            else:
                break
    else:
        # print('ARD starts after MTS')
        for i in range(len(sampleBx)):
            if sampleBx[i] <= minA:
                sampleBx[i] = -1
            else:
                break

    if maxA <= maxM:
        # print('MTS ends after ARD')
        for i in range(len(sampleBx)):
            if sampleBx[i] >= maxA:
                sampleBx[i] = -1
    else:
        # print('ARD ends after MTS')
        for i in range(len(sampleAx)):
            if sampleAx[i] >= maxM:
                sampleAx[i] = -1

    # removes all -1s
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

    # find the smaller sample
    new_len = min(len(sampleAx_clean), len(sampleBx_clean))

    sampleAresize_df = pd.DataFrame()
    sampleBresize_df = pd.DataFrame()

    sampleAresize_df[0] = sampleAx_clean
    sampleAresize_df[1] = sampleAy_clean
    sampleBresize_df[0] = sampleBx_clean
    sampleBresize_df[1] = sampleBy_clean

    holder_df = pd.DataFrame()
    holder_df[0] = lin_interpolate(sampleAresize_df[0], new_len)
    holder_df[1] = lin_interpolate(sampleAresize_df[1], new_len)
    holder_df[2] = lin_interpolate(sampleBresize_df[0], new_len)
    holder_df[3] = lin_interpolate(sampleBresize_df[1], new_len)

    return holder_df  # Ax, Ay, Bx, By


def lin_interpolate(
    array: np.ndarray, new_len: int
) -> np.ndarray:  # used in stretch data
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


# Output
def plot_two(
    ard, mts, ard_freq, avgArdError, ardError, errorBtw, address, ymax
):  # plots two curves and fills between
    if errorPlotBool != False:
        plt.plot(errorBtw[0], ardError, color="k", alpha=0.7, label=f"Error(t) [%]")
    plt.plot(
        mts[0], mts[1], color="r", linewidth=1.0, alpha=0.7, label=f"Applied MTS"
    )  # MTS load
    plt.plot(
        ard[0],
        ard[1],
        color="c",
        linewidth=0.8,
        label=f"Arduino ({int(avgArdError*10)/10}% off)",
    )  # Arduino load

    plt.minorticks_on()
    plt.grid(visible=None, which="major", linestyle="-", linewidth="1", axis="both")
    plt.grid(visible=None, which="minor", linestyle="--", linewidth="0.7", axis="y")

    plt.xlim(ard.iloc[0, 0], ard.iloc[len(ard) - 1, 0])
    plt.autoscale(False, axis="y")
    plt.ylim(0, ymax)
    plt.yticks(np.arange(0, ymax+1, step=int(ymax/100)*10))
    plt.legend()

    plt.title(f"{address[0]}\n{testInfo}")
    plt.title(f"@{int(ard_freq)}hz", loc="right", color="g")
    # plt.title(f'Circuit {circuitType}',loc='left') #TODO add both circut profiles if nessacary
    plt.xlabel("Time (hrs)")
    plt.ylabel("Load (kg)")

    plt.fill_between(
        errorBtw[0], errorBtw[1], errorBtw[2], color="gray", alpha=0.2, label="error"
    )

    plt.savefig(address[1])
    plt.cla()


def plot_three(
    rArd, cArd, mts, ard_freq, rAvgArdError, cAvgArdError, address, ymax
):  # plots three curves
    plt.plot(
        mts[0], mts[1], color="r", linewidth=1.0, alpha=0.7, label=f"Applied MTS"
    )  # MTS load
    plt.plot(
        rArd[0],
        rArd[1],
        color="c",
        linewidth=0.8,
        label=f"Raw ({int(rAvgArdError*10)/10}% off)",
    )  # raw arduino load
    plt.plot(
        cArd[0],
        cArd[1],
        color="k",
        linewidth=0.8,
        label=f"Corrected ({int(cAvgArdError*10)/10}% off)",
    )  # corrected arduino load

    plt.minorticks_on()
    plt.grid(visible=None, which="major", linestyle="-", linewidth="1", axis="both")
    plt.grid(visible=None, which="minor", linestyle="--", linewidth="0.7", axis="y")

    plt.xlim(rArd.iloc[0, 0], rArd.iloc[len(rArd) - 1, 0])
    plt.autoscale(False, axis="y")
    plt.ylim(0, ymax)
    plt.yticks(np.arange(0, ymax+1, step=int(ymax/100)*10))
    plt.legend()

    plt.title(f"{address[0]}\n{testInfo}")
    plt.title(f"@{int(ard_freq)}hz", loc="right", color="g")
    # plt.title(f'Circuit {circuitType}',loc='left') #TODO add both circut profiles if nessacary
    plt.xlabel("Time (hrs)")
    plt.ylabel("Load (kg)")

    plt.savefig(address[1])
    plt.cla()
    pass


def create_configure_save_dir():  # handles save location foler and copies datafiles
    # checks if arduino datafile exists
    if not os.path.isfile(arduinoDataFile):
        print(arduinoDataFile)
        print("Arduino datafile not found")
        exit(1)

    # checks if MTS datafile exists
    if not os.path.isfile(mtsDataFile):
        print("MTS datafile not found")
        exit(1)

    # copies datafile to save location
    if saveLocation != "":
        # creates save folder if necessary
        if not os.path.exists(saveLocation):
            os.makedirs(saveLocation)
            print(f'created "{saveLocation}" dir\n')

        try:
            shutil.copy(arduinoDataFile, saveLocation)
            os.rename(
                f"{saveLocation}/{arduinoDataFile}",
                f"{saveLocation}/DF Arduino ({testInfo}).txt",
            )
            print(
                f'Copied arduino datafile into "{saveLocation}" directory as "MTS Data File ({testInfo})"'
            )
        except shutil.SameFileError:
            print(f'Arduino datafile already exists in "{saveLocation}" directory')

        # copies mts file to datalocation
        try:
            shutil.copy(mtsDataFile, saveLocation)
            os.rename(
                f"{saveLocation}/{mtsDataFile}",
                f"{saveLocation}/DF MTS ({testInfo}).txt",
            )
            print(
                f'Copied MTS file into "{saveLocation}" directory as "MTS Data File ({testInfo})"\n'
            )
        except shutil.SameFileError:
            print(f'MTS file already exists in "{saveLocation}" directory\n')


def new_file_names(plot_type):  # creates chart titles and file names
    if plot_type == 'load_cell':
        # plot with raw with mts data
        rawMts = [f"Raw Arduino Data vs MTS Data (cell frame)", " "]
        # plot with fixed and mts data
        fixedMts = [f"Corrected Arduino Data vs MTS Data (cell frame)", " "]
        # plot with raw, fixed, and mts data
        rawFixedMts = [f"Raw and Corrected Arduino Data vs MTS Data (cell frame)", " "]

    if plot_type == 'block':
        # plot with raw with mts data
        rawMts = [f"Raw Arduino Data vs MTS Data (block frame)", " "]
        # plot with fixed and mts data
        fixedMts = [f"Corrected Arduino Data vs MTS Data (block frame)", " "]
        # plot with raw, fixed, and mts data
        rawFixedMts = [f"Raw and Corrected Arduino Data vs MTS Data (block frame)", " "]

    if plot_type == 'rearAxle':
        # plot with raw with mts data
        rawMts = [f"Raw Arduino Data vs MTS Data (rearAxle frame)", " "]
        # plot with fixed and mts data
        fixedMts = [f"Corrected Arduino Data vs MTS Data (rearAxle frame)", " "]
        # plot with raw, fixed, and mts data
        rawFixedMts = [f"Raw and Corrected Arduino Data vs MTS Data (rearAxle frame)", " "]


    if saveLocation == "":
        rawMts[1] = f"PLT {rawMts[0]} ({testInfo}).png"
        fixedMts[1] = f"PLT {fixedMts[0]} ({testInfo}).png"
        rawFixedMts[1] = f"PLT {rawFixedMts[0]} ({testInfo}).png"
    else:
        rawMts[1] = f"{saveLocation}/PLT {rawMts[0]} ({testInfo}).png"
        fixedMts[1] = f"{saveLocation}/PLT {fixedMts[0]} ({testInfo}).png"
        rawFixedMts[1] = f"{saveLocation}/PLT {rawFixedMts[0]} ({testInfo}).png"

    return rawMts, fixedMts, rawFixedMts


def write_info_file():
    if saveLocation != "":
        with open(f"{saveLocation}/info ({testInfo}).txt", "w") as f:
            f.write(f"created on {date.today()}\n\n")
            f.write(f"ADC equation: [kg] = {adcSlope}*x + {adcOffset}\n\n")
            f.write(f'Scale factor: {scale}\n\n')
    else:
        with open(f"info ({testInfo}).txt", "w") as f:
            f.write(f"created on {date.today()}\n\n")
            f.write(f"ADC equation: [kg] = {adcSlope}*x + {adcOffset}\n\n")
            f.write(f'Scale factor: {scale}\n\n')


@Gooey(
    advanced=True,
    default_size=(800, 600),  # window size
    sidebar_title=f"Select Sensor",  # titles sidebar
    tabbed_groups=True,  # enables tabs
    navigation="Tabbed",  # sets groups
    disable_progress_bar_animation=True,  # hide progress bar
    body_bg_color="#d0c792",  # color
    header_bg_color="#cccaac",
    # footer_bg_color = '#c6c7be',
    # sidebar_bg_color = '#c6c7be',
    # terminal_panel_color = '#cccaac',
)
def main_argparse():  # main GUI (1st screen)
    currentFileLocation = os.getcwd()
    currentFolder = currentFileLocation.split("/")[-1]
    parentFolder = currentFileLocation.split("/")[-2]
    topTwoPath = f"{parentFolder}/{currentFolder}"

    # argument parser
    parser = GooeyParser()

    """
    gooey sidebar setup
    """
    subs = parser.add_subparsers()
    flexi_parser = subs.add_parser(
        "FlexiForce"
    )  # help='Configure the Flexi Force paramaters'
    tallLC_parser = subs.add_parser(
        "TallLoadCell"
    )  # help='Configure the tall load cell paramaters'
    buttonLC_parser = subs.add_parser(
        "ButtonLoadCell"
    )  # help='Configure the button load cell paramaters'

    """
    flexi force side bar
    """
    # tab setup
    general_parser = flexi_parser.add_argument_group(
        "General"
    )  # help='General configuration'
    plot_parser = flexi_parser.add_argument_group(
        "Plot config"
    )  # help='Configure the plot'
    mts_tune_parser = flexi_parser.add_argument_group(
        "MTS Tuning"
    )  # help='Tune the MTS data paramaters'
    flexi_config_parser = flexi_parser.add_argument_group(
        "FlexiForce Config"
    )  # help='Tune the MTS data paramaters'

    # gooey tabs
    # arduino datafile
    general_parser.add_argument(  # Arduino datafile location
        "--ard",
        dest="Arduino",
        help="Arduino datafile location  (.TXT)",
        default="DF Arduino",
    )  # log6/LOG_6.txt
    general_parser.add_argument(  # Arduino datafile delimiter
        "--ardSep",
        dest="ArduinoSep",
        help="Arduino file delimiter (default: TAB)",
        default="\t",
        type=str,
    )
    # MTS datafile
    general_parser.add_argument(  # MTS datafile location
        "--mts", dest="mts", help="MTS datafile location (.txt)", default="DF MTS"
    )  # log6/hy_test1_glue6.txt
    general_parser.add_argument(  # MTS datafile delimiter
        "--mtsSep",
        dest="mtsSep",
        help="MTS file delimiter (default: TAB)",
        default="\t",
    )
    # saving
    general_parser.add_argument(  # save location
        "--saveLocation",
        dest="saveLocation",
        help=f"Save folder ({topTwoPath}/___)",
        default="",
    )  # log6/
    general_parser.add_argument(  # test info
        "--testInfo", dest="testInfo", help="Test info", default="script testing"
    )
    
    # MTS tuning
    # time shift
    mts_tune_parser.add_argument(  # MTS time shift
        "--mtsShift",
        dest="mtsShift",
        help="MTS time shift (in seconds)",
        default="0.00",
        type=float,
    )
    # block specs
    mts_tune_parser.add_argument(  # Block length and width
        "--blockArea",
        dest="blockArea",
        help="Block length and width (in)",
        nargs="+",
        default=[4.5, 1.375],
        type=float,
    )
    
    # plot load cell frame 
    plot_parser.add_argument(   # load cell frame
        "--loadCellFrame",
        dest="loadCellFrame",
        help="Plots load in the load cellframe",
        action="store_true",
        default=True,
    )
    # plot block frame 
    plot_parser.add_argument(   # block frame
        "--blockFrame",
        dest="blockFrame",
        help="Plots load in the block frame",
        action="store_true",
    )
    # plot rearAxle frame 
    plot_parser.add_argument(   # rear axel frame
        "--rearAxleFrame",
        dest="rearAxleFrame",
        help="Plots load in the rear axle frame",
        action="store_true",
    )

    # plot error trend over data
    plot_parser.add_argument(
        "--errorPlot",
        dest="errorPlot",
        help="Plots error over load readings",
        action="store_true",
    )
    
    #slope
    flexi_config_parser.add_argument(  # ADC slope
        "--slope",
        dest="slope",
        help=f"ADC slope \n(y = 'm'*x + b)",
        default="0.0105",
        type=float,
    )
    flexi_config_parser.add_argument(  # ADC offset
        "--offset",
        dest="offset",
        help=f"ADC offset \n(y = m*x + 'b')",
        default="-105.0",
        type=float,
    )
    # sensor area
    flexi_config_parser.add_argument(  # FlexiForce area
        "--sensorArea",
        dest="sensorArea",
        help="FlexiForce sensor area (in^2)",
        default="0.785852",
        type=float,
    )
    # derv
    flexi_config_parser.add_argument(  # derv threshold
        "--dervThresh",
        dest="dervThresh",
        help=f"Used to fix ADC data",
        default="20",
        type=float,
    )
    # scale factor (default n = 1) 
    flexi_config_parser.add_argument(  # button load cell area
        "--scale",
        dest="scale",
        help="Scale input by",
        default="1",
        type=float,
    )
    

    """
    tall load cell side bar
    """
    # tab setup
    general_parser = tallLC_parser.add_argument_group(
        "General"
    )  # help='General configuration'
    plot_parser = tallLC_parser.add_argument_group(
        "Plot config"
    )  # help='Configure the plot'
    mts_tune_parser = tallLC_parser.add_argument_group(
        "MTS Tuning"
    )  # help='Tune the MTS data paramaters'
    tallLC_config_parser = tallLC_parser.add_argument_group(
        "Tall Load Cell config"
    )  # help='Tune the MTS data paramaters'

    # gooey tabs
    # arduino datafile
    general_parser.add_argument(  # Arduino datafile location
        "--ard",
        dest="Arduino",
        help="Arduino datafile location  (.TXT)",
        default="DF Arduino",
    )  # log6/LOG_6.txt
    general_parser.add_argument(  # Arduino datafile delimiter
        "--ardSep",
        dest="ArduinoSep",
        help="Arduino file delimiter (default: TAB)",
        default="\t",
        type=str,
    )
    # MTS datafile
    general_parser.add_argument(  # MTS datafile location
        "--mts", dest="mts", help="MTS datafile location (.txt)", default="DF MTS"
    )  # log6/hy_test1_glue6.txt
    general_parser.add_argument(  # MTS datafile delimiter
        "--mtsSep",
        dest="mtsSep",
        help="MTS file delimiter (default: TAB)",
        default="\t",
    )
    # saving
    general_parser.add_argument(  # save location
        "--saveLocation",
        dest="saveLocation",
        help=f"Save folder ({topTwoPath}/___)",
        default="",
    )  # log6/
    general_parser.add_argument(  # test info
        "--testInfo", dest="testInfo", help="Test info", default="script testing"
    )
    
    # MTS tuning
    # time shift
    mts_tune_parser.add_argument(  # MTS time shift
        "--mtsShift",
        dest="mtsShift",
        help="MTS time shift (in seconds)",
        default="0.00",
        type=float,
    )
    # block specs
    mts_tune_parser.add_argument(  # Block length and width
        "--blockArea",
        dest="blockArea",
        help="Block length and width (in)",
        nargs="+",
        default=[4.29, 1.92],
        type=float,
    )

    # plot load cell frame 
    plot_parser.add_argument(   # load cell frame
        "--loadCellFrame",
        dest="loadCellFrame",
        help="Plots load in the load cellframe",
        action="store_true",
        default=True,
    )
    # plot block frame 
    plot_parser.add_argument(   # block frame
        "--blockFrame",
        dest="blockFrame",
        help="Plots load in the block frame",
        action="store_true",
    )
    # plot rearAxle frame 
    plot_parser.add_argument(   # rear axel frame
        "--rearAxleFrame",
        dest="rearAxleFrame",
        help="Plots load in the rear axle frame",
        action="store_true",
    )

    # plot error trend over data
    plot_parser.add_argument(   # error plot
        "--errorPlot",
        dest="errorPlot",
        help="Plots error over load readings",
        action="store_true",
    )
    # derv threshold
    plot_parser.add_argument(   # derv threshold
        "--dervThresh",
        dest="dervThresh",
        help="leave unchecked \n(for flexi sensor only)",
        default=False,
        action="store_false",
    )
    
    # slope
    tallLC_config_parser.add_argument(  # tall load cell slope
        "--slope",
        dest="slope",
        help=f"Tall load cell slope \n(y = 'm'*x + b)",
        default="1",
        type=float,
    )
    tallLC_config_parser.add_argument(  # tall load cell offset
        "--offset",
        dest="offset",
        help=f"Tall load cell offset \n(y = m*x + 'b')",
        default="0",
        type=float,
    )
    # sensor area
    tallLC_config_parser.add_argument(  # tall load cell area
        "--sensorArea",
        dest="sensorArea",
        help="Tall load cell sensor area (in^2)",
        default="0.762",
        type=float,
    )
    # scale factor (default n = 1) 
    tallLC_config_parser.add_argument(  # button load cell area
        "--scale",
        dest="scale",
        help="Scale input by",
        default="1",
        type=float,
    )
    
    
    """
    button load cell side bar
    """
    # tab setup
    general_parser = buttonLC_parser.add_argument_group(
        "General"
    )  # help='General configuration'
    plot_parser = buttonLC_parser.add_argument_group(
        "Plot config"
    )  # help='Configure the plot'
    mts_tune_parser = buttonLC_parser.add_argument_group(
        "MTS Tuning"
    )  # help='Tune the MTS data paramaters'
    buttonLC_config_parser = buttonLC_parser.add_argument_group(
        "Button Load Cell config"
    )  # help='Tune the MTS data paramaters'

    # gooey tabs
    # arduino datafile
    general_parser.add_argument(  # Arduino datafile location
        "--ard",
        dest="Arduino",
        help="Arduino datafile location  (.TXT)",
        default="DF Arduino",
    )  # log6/LOG_6.txt
    general_parser.add_argument(  # Arduino datafile delimiter
        "--ardSep",
        dest="ArduinoSep",
        help="Arduino file delimiter (default: TAB)",
        default="\t",
        type=str,
    )
    # MTS datafile
    general_parser.add_argument(  # MTS datafile location
        "--mts", dest="mts", help="MTS datafile location (.txt)", default="DF MTS"
    )  # log6/hy_test1_glue6.txt
    general_parser.add_argument(  # MTS datafile delimiter
        "--mtsSep",
        dest="mtsSep",
        help="MTS file delimiter (default: TAB)",
        default="\t",
    )
    # saving
    general_parser.add_argument(  # save location
        "--saveLocation",
        dest="saveLocation",
        help=f"Save folder ({topTwoPath}/___)",
        default="",
    )  # log6/
    general_parser.add_argument(  # test info
        "--testInfo", dest="testInfo", help="Test info", default="script testing"
    )
    
    # MTS tuning
    # time shift
    mts_tune_parser.add_argument(  # MTS time shift
        "--mtsShift",
        dest="mtsShift",
        help="MTS time shift (in seconds)",
        default="0.00",
        type=float,
    )
    # block specs
    mts_tune_parser.add_argument(  # Block length and width
        "--blockArea",
        dest="blockArea",
        help="Block length and width (in)",
        nargs="+",
        default=[4.25, 1.94],
        type=float,
    )
    
    # plot load cell frame 
    plot_parser.add_argument(   # load cell frame
        "--loadCellFrame",
        dest="loadCellFrame",
        help="Plots load in the load cellframe",
        action="store_true",
    )
    # plot block frame 
    plot_parser.add_argument(   # block frame
        "--blockFrame",
        dest="blockFrame",
        help="Plots load in the block frame",
        action="store_true",
    )
    # plot rearAxle frame 
    plot_parser.add_argument(   # rear axel frame
        "--rearAxleFrame",
        dest="rearAxleFrame",
        help="Plots load in the rear axle frame",
        action="store_true",
    )
    # plot error trend over data
    plot_parser.add_argument(   # error plot
        "--errorPlot",
        dest="errorPlot",
        help="Plots error over load readings",
        action="store_true",
    )
    # derv threshold
    plot_parser.add_argument(   # derv threshold
        "--dervThresh",
        dest="dervThresh",
        help="leave unchecked \n(for flexi sensor only)",
        default=False,
        action="store_false",
    )
    
    # slope
    buttonLC_config_parser.add_argument(  # button load cell slope
        "--slope",
        dest="slope",
        help=f"Button load cell slope \n(y = 'm'*x + b)",
        default="1",
        type=float,
    )
    buttonLC_config_parser.add_argument(  # button load cell offset
        "--offset",
        dest="offset",
        help=f"Button load cell offset \n(y = m*x + 'b')",
        default="0",
        type=float,
    )
    # sensor area
    buttonLC_config_parser.add_argument(  # button load cell area
        "--sensorArea",
        dest="sensorArea",
        help="Button load cell sensor area (in^2)",
        default="0.486",
        type=float,
    ) 
    # scale factor (default n = 1) 
    buttonLC_config_parser.add_argument(  # button load cell area
        "--scale",
        dest="scale",
        help="Scale input by",
        default="1",
        type=float,
    )
   

    """
    parse arguments
    """
    args = parser.parse_args()

    # from gooey tab - general
    # arduino
    arduinoDataFile = f"{args.Arduino}.TXT" # TODO may need to be .txt
    ardSepBy = args.ArduinoSep
    # MTS
    mtsDataFile = f"{args.mts}.txt"
    mtsSepBy = args.mtsSep
    # saving
    testInfo = args.testInfo
    saveLocation = args.saveLocation

    # from gooey tab - mts
    mtsTimeShift = args.mtsShift
    blockLength, blockWidth = args.blockArea

    # from gooey tab - plot
    loadCellFramePlot = args.loadCellFrame
    blockFramePlot = args.blockFrame
    rearAxleFramePlot = args.rearAxleFrame

    errorPlotBool = args.errorPlot
    dervThresh = args.dervThresh

    slope = args.slope
    offset = args.offset
    sensorArea = args.sensorArea
    scale = args.scale

    return (
        # saving
        currentFolder,
        # arduino data
        arduinoDataFile,
        ardSepBy,
        # mts data
        mtsDataFile,
        mtsSepBy,
        mtsTimeShift,
        # block
        blockLength,
        blockWidth,
        # sensor
        sensorArea,
        # saving
        testInfo,
        saveLocation,
        # scaling
        slope,
        offset,
        # correcting arduino
        dervThresh,
        # plotting
        errorPlotBool,
        scale,
        # which plots
        loadCellFramePlot,
        blockFramePlot,
        rearAxleFramePlot,
    )


if __name__ == "__main__":
    print("- - - - -  LoadAlgorythem2.0.py  - - - - -")
    (  # makes vars from argparse global
        # saving
        currentFolder,
        # arduino data
        arduinoDataFile,
        ardSepBy,
        # mts data
        mtsDataFile,
        mtsSepBy,
        mtsTimeShift,
        # block
        blockLength,
        blockWidth,
        # sensor
        sensorArea,
        # saving
        testInfo,
        saveLocation,
        # scaling
        adcSlope,
        adcOffset,
        # correcting arduino
        dervThresh,
        # plotting
        errorPlotBool,
        scale,
        # which plots
        loadCellFramePlot,
        blockFramePlot,
        rearAxleFramePlot,
    ) = main_argparse()

    create_configure_save_dir()  # creates and configures save directories
    main_process()  # main process
    write_info_file()  # write info file
    print("- - - - - - - - - - - - - - - - - - - - -")
