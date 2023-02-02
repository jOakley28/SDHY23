# SDHY23
Senior Design - Hyster Yale Axial Load Monitoring

# Post-Processing Algorithm for Iterative Testing
## [LoadAlgorithm  v2.0 (WIP)](https://github.com/jOakley28/SDHY23/blob/main/LoadAlgorithm1.16.py)
* Time non-linearity correction
* Raw ADC to load (kg) conversion 
* Refined error calculation
* Visual representations of prosessed siginal 

## [LoadAlgorithm v1.16](https://github.com/jOakley28/SDHY23/blob/main/LoadAlgorithm1.16.py)
* Non-linear ADC to load (kg) conversion
* Robust plotting
* Noise reduction (fft)
* Redumentry error calculation

# Pre-Processing Algorithm (deployed to hardware)
## Arduino v2.0
* Conducts onboard high and low pass filtering

## Arduino v1.0
* Collects and writes raw to CSV for post-processing
