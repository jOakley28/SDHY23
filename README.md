# SDHY23
Senior Design - Hyster Yale Axial Load Monitoring

# Post-Processing Algorithm for Iterative Testing
## [LoadAlgorithm  v2.0 (WIP)](https://github.com/jOakley28/SDHY23/blob/main/LoadAlgorithm1.16.py)
* Time non-linearity correction
* Raw ADC to load (kg) conversion 
* Refined error calculation
* Visual representations of prosessed siginal 
* GUI

## [LoadAlgorithm v1.16](https://github.com/jOakley28/SDHY23/blob/main/LoadAlgorithm1.16.py)
* Non-linear ADC to load (kg) conversion
* Robust plotting
* Noise reduction (fft)
* Redumentry error calculation

# Pre-Processing Algorithm (deployed to hardware)
## [SDHY_Load_Cell_Datalogger](https://github.com/jOakley28/SDHY23/tree/main/SDHY_Load_Cell_Datalogger/SDHY_Load_Cell_Datalogger.ino)
* Uses an Arduino Uno and HX711 Wheatstone bridge and ADC module
* Outputs load in kg

## Arduino v1.0
* Collects and writes raw to CSV for post-processing
