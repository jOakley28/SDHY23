""" LoadAlgorithm2.0
    
    Senior Design (SDHY) 2022-2023
    George Fox University 

    Jacob Hankland
    jhankland19@georgefox.edu
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
import gooey


# SETUP
plt.rcParams["figure.figsize"] = [20, 12]  # sets figure size
plt.rcParams.update({"font.size": 18})  # sets font size
warnings.filterwarnings("ignore")  # hide warnings


# main process
def main_process():
    # handle raw ADC data
    rawAdcData = pd.DataFrame()
    rawAdcData, rawAdcFreq = import_arduino_data(
        arduinoDataFile, ardSepBy
    )  # import arduino data

    # TODO look for change in adc
    correctedAdcData = detect_change_in_load(rawAdcData)
    correctedAdcData[1] = convert_adc_to_kg(correctedAdcData[1])  # converts ADC to kg

    # handle mts data
    mtsData = pd.DataFrame()
    mtsData, mtsFreq = import_mts_data(
        mtsDataFile, mtsSepBy, mtsTimeShift
    )  # imports mts data
    mtsData[1] = correct_mts_data(mtsData[1])  # corrects mts (-N to +kg) (area ratio)

    # calculate percent error
    print("Estimating error...")
    rawAdcDataCopy = rawAdcData.copy()
    correctedAdcDataCopy = correctedAdcData.copy()
    mtsDataCopy = mtsData.copy()
    rAdcMtsError, rAvgError, rEbtw = errorCalculation(
        rawAdcData, mtsData
    )  # raw ADC vs MTS precent error
    cAdcMtsError, cAvgError, cEbtw = errorCalculation(
        correctedAdcData, mtsData
    )  # corrected ADC vs MTS precent error

    # plot data
    print("Plotting data...")
    rawMts_str, correctedMts_str, rawcorrectedMts_str = file_names()
    plot_two(
        rawAdcDataCopy,
        mtsDataCopy,
        rawAdcFreq,
        rAvgError,
        rAdcMtsError,
        rEbtw,
        rawMts_str,
    )  # plot raw ADC and MTS
    plot_two(
        correctedAdcDataCopy,
        mtsDataCopy,
        rawAdcFreq,
        cAvgError,
        cAdcMtsError,
        cEbtw,
        correctedMts_str,
    )  # plot corrected ADC and MTS
    # plot_three() #TODO
    print("Plotting complete")


# Import data
def import_arduino_data(arduinoDataFile, ardSepBy):  # get arduino data from CSV file
    print("Getting arduino data...")
    # put the data into a pandas dataframe
    arduino_data_df = pd.read_csv(
        arduinoDataFile, sep=ardSepBy, header=None, low_memory=False
    )

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
    rawMtsData_df = rawMtsData
    blockArea = blockLength * blockWidth  # difined by user (in^2)
    sensorArea = 0.785852  # from data sheet (in^2)
    areaRatio = blockArea / sensorArea
    nToKg = -1 / 9.806  # downward force (N) (in the negative direction) to load (kg)
    corFactor = nToKg / areaRatio
    correctedMtsData = rawMtsData.apply(lambda x: x * corFactor)
    return correctedMtsData


# TODO Determine if there was a change in loading
def detect_change_in_load(data):  # determine if there was a change in loading
    return data


def take_first_derivative(time, data):  # takes the first derivative of a dataset
    dataFirstDir = [0]  # creates array ofset by 1 to account for 1st dir offset
    for i in range(
        len(data) - 1
    ):  # takes the difrence between each two data points and divides by the change in time
        ithDerivative = (data[i + 1] - data[i]) / (time[i + 1] - time[i])
        dataFirstDir = dataFirstDir.append(
            ithDerivative
        )  # appends the derivative array with the next ith derivative
    return dataFirstDir


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
    # adcDataKg = pd.DataFrame()
    adcDataKg = adcData

    [m, b] = [0.007726, -77.5564]  # circut B profile
    adcDataKg = adcDataKg.apply(lambda x: m * x + b)
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
    ard, mts, ard_freq, avgArdError, ardError, errorBtw, address
):  # plots two curves and fills between
    plt.plot(
        mts[0], mts[1], color="r", linewidth=1.0, alpha=0.7, label=f"Applied MTS"
    )  # MTS load
    plt.plot(
        ard[0],
        ard[1],
        color="c",
        linewidth=0.8,
        label=f"Raw ({int(avgArdError*10)/10}%)",
    )  # raw load

    plt.minorticks_on()
    plt.grid(visible=None, which="major", linestyle="-", linewidth="1", axis="both")
    plt.grid(visible=None, which="minor", linestyle="--", linewidth="0.7", axis="y")

    plt.xlim(ard.iloc[0, 0], ard.iloc[len(ard) - 1, 0])
    plt.autoscale(False, axis="y")
    plt.ylim(0, 110)
    plt.yticks(np.arange(0, 111, step=10))
    plt.legend()

    plt.title(f"{address[0]}\n{testInfo}")
    plt.title(f"@{int(ard_freq)}hz - latency* 0s", loc="right", color="g")
    # plt.title(f'Circuit {circuitType}',loc='left') #TODO add both circut profiles if nessacary
    plt.xlabel("Time (hrs)")
    plt.ylabel("Load (kg)")

    plt.fill_between(
        errorBtw[0], errorBtw[1], errorBtw[2], color="gray", alpha=0.2, label="error"
    )
    plt.savefig(address[1])
    plt.cla()


def plot_three():  # TODO plots three curves
    pass


def create_configure_save_dir():  # handles save location foler and copies datafiles
    # creates save folder if necessary
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
        print(f'created "{saveLocation}" dir\n')

    # checks if arduino datafile exists
    if not os.path.isfile(arduinoDataFile):
        print("Data file not found")
        exit(1)

    # checks if MTS datafile exists
    if not os.path.isfile(mtsDataFile):
        print("MTS file not found")
        exit(1)

    # copies datafile to save location
    try:
        shutil.copy(arduinoDataFile, saveLocation)
        print(f'Copied arduino datafile into "{saveLocation}" directory')
    except shutil.SameFileError:
        print(f'Arduino datafile already exists in "{saveLocation}" directory')

    # copies mts file to datalocation
    try:
        shutil.copy(mtsDataFile, saveLocation)
        print(f'Copied MTS file into "{saveLocation}" directory\n')
    except shutil.SameFileError:
        print(f'MTS file already exists in "{saveLocation}" directory\n')

    pass


def file_names():  # creates chart titles and file names
    # plot with raw with mts data
    rawMts = [f"raw arduino over mts data", " "]
    rawMts[1] = f"{saveLocation}/PLT {rawMts[0]}.png"

    # plot with fixed and mts data
    fixedMts = [f"corrected arduino over mts data", " "]
    fixedMts[1] = f"{saveLocation}/PLT {fixedMts[0]}.png"

    # plot with raw, fixed, and mts data
    rawFixedMts = [f"raw and corrected arduino over mts data", " "]
    rawFixedMts[1] = f"{saveLocation}/PLT {rawFixedMts[0]}.png"

    return rawMts, fixedMts, rawFixedMts


if __name__ == "__main__":
    # TODO add argument parser

    # arduino
    arduinoDataFile = "log6/LOG_6.txt"
    ardSepBy = "\t"

    # MTS
    mtsDataFile = "log6/hy_test1_glue6.txt"
    mtsSepBy = "\t"
    mtsTimeShift = 0.00  # in seconds
    blockLength = 2.5  # in inches
    blockWidth = 2.5  # in inches

    # saving
    testInfo = "building algorithm"
    saveLocation = "log6"

    create_configure_save_dir()  # creates and configures save directories
    main_process()  # main process
