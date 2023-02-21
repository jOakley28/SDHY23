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
from gooey import Gooey


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

    # detect change in load based on ADC data
    rawAdcDataCopy = (
        rawAdcData.copy()
    )  # creates a copy of raw ADC data for error calculation
    correctedAdcData = detect_change_in_load(rawAdcDataCopy)  # looks for change in load
    correctedAdcDataCopy = correctedAdcData.copy()  # copy for error calculation

    # convert ADC value to kg
    rawAdcData[1] = convert_adc_to_kg(rawAdcData[1])
    correctedAdcData[1] = convert_adc_to_kg(correctedAdcData[1])

    # handle mts data
    mtsData = pd.DataFrame()
    mtsData, mtsFreq = import_mts_data(
        mtsDataFile, mtsSepBy, mtsTimeShift
    )  # imports mts data
    mtsData[1] = correct_mts_data(mtsData[1])  # corrects mts (-N to +kg) (area ratio)
    mtsDataCopy = mtsData.copy()  # creates a copy of mts data for error calculation

    # convert ADC value to kg
    rawAdcDataCopy[1] = convert_adc_to_kg(rawAdcDataCopy[1])
    correctedAdcDataCopy[1] = convert_adc_to_kg(correctedAdcDataCopy[1])

    # calculate percent error
    print("Estimating error...\n")
    rAdcMtsError, rAvgError, rEbtw = errorCalculation(
        rawAdcDataCopy, mtsDataCopy
    )  # raw ADC vs MTS precent error
    cAdcMtsError, cAvgError, cEbtw = errorCalculation(
        correctedAdcDataCopy, mtsDataCopy
    )  # corrected ADC vs MTS precent error

    # plot data
    print("Plotting data...")
    rawMts_str, correctedMts_str, rawcorrectedMts_str = new_file_names()
    plot_two(
        rawAdcData,
        mtsData,
        rawAdcFreq,
        rAvgError,
        rAdcMtsError,
        rEbtw,
        rawMts_str,
    )  # plot raw ADC and MTS

    plot_two(
        correctedAdcData,
        mtsData,
        rawAdcFreq,
        cAvgError,
        cAdcMtsError,
        cEbtw,
        correctedMts_str,
    )  # plot corrected ADC and MTS

    plot_three(
        rawAdcData,
        correctedAdcData,
        mtsDataCopy,
        rawAdcFreq,
        rAvgError,
        cAvgError,
        rawcorrectedMts_str,
    )  # plot raw and corrected ADC along with MTS

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
    dataToDetect = data.copy()
    detectedAdc = pd.DataFrame()
    detectedAdc[0] = data[0]
    adc_corrected = []

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
    ard, mts, ard_freq, avgArdError, ardError, errorBtw, address
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
    plt.ylim(0, 110)
    plt.yticks(np.arange(0, 111, step=10))
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
    rArd, cArd, mts, ard_freq, rAvgArdError, cAvgArdError, address
):  # TODO plots three curves
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
    plt.ylim(0, 110)
    plt.yticks(np.arange(0, 111, step=10))
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


def new_file_names():  # creates chart titles and file names
    # plot with raw with mts data
    rawMts = [f"Raw Arduino Data vs MTS Data", " "]
    # plot with fixed and mts data
    fixedMts = [f"Corrected Arduino Data vs MTS Data", " "]
    # plot with raw, fixed, and mts data
    rawFixedMts = [f"Raw and Corrected Arduino Data vs MTS Data", " "]

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
            f.write(f"ADC equation: [kg] = {adcSlope}*x + {adcOffset}")
    else:
        with open(f"info ({testInfo}).txt", "w") as f:
            f.write(f"created on {date.today()}\n\n")
            f.write(f"ADC equation: [kg] = {adcSlope}*x + {adcOffset}")


@Gooey(advanced=True, default_size=(730, 840))
def main_argparse():  # main GUI (1st screen)
    currentFileLocation = os.getcwd()
    currentFolder = currentFileLocation.split("/")[-1]
    parentFolder = currentFileLocation.split("/")[-2]
    topTwoPath = f"{parentFolder}/{currentFolder}"

    # argument parser
    parser = ArgumentParser()
    # sensor type dropdown
    parser.add_argument(
        dest="sensorType",
        default="FlexiForce",
        const="FlexiForce",
        nargs="?",
        help="The type of sensor used in this test",
        choices=["FlexiForce", "Button Loadcell", "Tall Loadcell"],
    )
    # arduino datafile
    parser.add_argument(  # Arduino datafile location
        "-a",
        "--ard",
        dest="Arduino",
        help="Arduino datafile location  (.TXT)",
        default="DF Arduino",
    )  # log6/LOG_6.txt
    parser.add_argument(  # Arduino datafile delimiter
        "-d",
        "--ardSep",
        dest="ArduinoSep",
        help="Arduino datafile delimiter",
        default="\t",
        type=str,
    )
    # MTS datafile
    parser.add_argument(  # MTS datafile location
        "-m", "--mts", dest="MTS", help="MTS datafile location (.txt)", default="DF MTS"
    )  # log6/hy_test1_glue6.txt
    parser.add_argument(  # MTS datafile delimiter
        "-t",
        "--mtsSep",
        dest="mtsSep",
        help="MTS datafile delimiter",
        default="\t",
    )
    # saving
    parser.add_argument(  # save location
        "-l",
        "--saveLocation",
        dest="saveLocation",
        help=f"Save folder \n({topTwoPath}/___)",
        default="",
    )  # log6/
    parser.add_argument(  # test info
        "-i", "--testInfo", dest="testInfo", help="Test info", default="script testing"
    )
    # plot error trend over data
    parser.add_argument(
        "-E",
        "--errorPlot",
        dest="errorPlot",
        help="Plots error over load readings",
        action="store_true",
    )
    # tuning
    parser.add_argument(  # MTS time shift
        "-s",
        "--mtsShift",
        dest="mtsShift",
        help="MTS time shift (in seconds)",
        default="0.00",
        type=float,
    )
    parser.add_argument(  # Block length and width
        "-A",
        "--blockArea",
        dest="blockArea",
        help="Block length and width (in inches)",
        nargs="+",
        default=[4.5, 1.375],
        type=float,
    )
    # lin eq
    parser.add_argument(  # ADC slope
        "-x",
        "--adcSlope",
        dest="adcSlope",
        help=f"ADC slope \n(y = 'm'*x + b)",
        default="0.0105",
        type=float,
    )
    parser.add_argument(  # ADC offset
        "-b",
        "--adcOffset",
        dest="adcOffset",
        help=f"ADC offset \n(y = m*x + 'b')",
        default="-105.0",
        type=float,
    )
    # derv
    parser.add_argument(  # derv threshold
        "-D",
        "--dervThresh",
        dest="dervThresh",
        help=f"Derivative threshold used to fix ADC data",
        default="20",
        type=float,
    )

    args = parser.parse_args()

    # sensor type
    sensorType = args.sensorType
    # arduino
    arduinoDataFile = f"{args.Arduino}.TXT"
    print(arduinoDataFile)
    ardSepBy = args.ArduinoSep
    # MTS
    mtsDataFile = f"{args.MTS}.txt"
    mtsSepBy = args.mtsSep
    mtsTimeShift = args.mtsShift
    blockLength, blockWidth = args.blockArea
    # saving
    testInfo = args.testInfo
    saveLocation = args.saveLocation
    # plotting
    errorPlotBool = args.errorPlot

    adcSlope = args.adcSlope
    adcOffset = args.adcOffset
    dervThresh = args.dervThresh

    return (
        currentFolder,
        arduinoDataFile,
        ardSepBy,
        mtsDataFile,
        mtsSepBy,
        mtsTimeShift,
        blockLength,
        blockWidth,
        testInfo,
        saveLocation,
        adcSlope,
        adcOffset,
        dervThresh,
        errorPlotBool,
    )


def flexi_argparse():
    pass


def buttoncell__argparse():
    pass


def tallcell_argparse():
    pass


if __name__ == "__main__":
    print("- - - - -  LoadAlgorythem2.0.py  - - - - -")
    (  # makes vars from argparse global
        currentFolder,
        arduinoDataFile,
        ardSepBy,
        mtsDataFile,
        mtsSepBy,
        mtsTimeShift,
        blockLength,
        blockWidth,
        testInfo,
        saveLocation,
        adcSlope,
        adcOffset,
        dervThresh,
        errorPlotBool,
    ) = main_argparse()

    create_configure_save_dir()  # creates and configures save directories
    main_process()  # main process
    write_info_file()  # write info file
    print("- - - - - - - - - - - - - - - - - - - - -")
