# Invasive Species Detection and Monitoring

Invasive species pose a global threat to ecosystem health and economic stability. Accurate information on the distribution and abundance of invasive species is crucial for meeting global biodiversity conservation goals and reducing the costs of invasion. Traditional methods of collecting data can be both labor-intensive and prohibitively expensive, limiting their utility in addressing the global proliferation of invasive species. Thus, the escalating scale and impacts of invasive species necessitate the development of alternative approaches for efficient detection and dynamic monitoring.

# Problem Statement

The purpose of this project is to address the challenges associated with invasive species management and prediction by developing innovative techniques. Species Distribution Models (SDMs) are the backbone of invasive species management and prediction but are not without flaws and bias. Remote sensing offers a promising alternative to traditional on-the-ground data collection for detecting and monitoring plant species and communities, as it overcomes logistical constraints and provides large-scale, cost-effective data. However, species-level detection can be challenging due to factors such as population density, growth form, and phenology.

# Solution

This project aims to compare species distribution models and remote sensing applications for species mapping to assess and monitor invasive species' distributions. Remote sensing techniques aim to predict actual species distribution with a higher resolution, whereas SDMs primarily focus on broad-scale climatic suitability models for potential species distribution. The complementary nature of SDMs and remote sensing offers unique insights and applications for invasive species detection and management.

# Contents

- The `/leafy-spurge-demography` directory contains data, scripts/code, models for mapping and monitoring the invasive plant species, Leafy spurge. Author: Thomas Lake; 2022-2023. [Repository Link](https://github.com/lake-thomas/landsat-leafy-spurge)

- The `/analysesNotebooks` directory contains Python development code in Jupyter Notebooks used for data processing, deep learning model development, and predictions. Code can be interactively run for diagnostics.
    - `TemporalCNN_GenerateTrainingData_Points_Gcloud.ipynb` contains Google Earth Engine Python API code to sample Landsat satellite band data from labeled points. Points contain latitude/longitude location information and a land cover class description from the National Land Cover Database. Landsat data samples are arranged into three-year sequences (e.g., one point contains spectral data from 2018-2019-2020). 
    - `TemporalCNN_GenerateImages_Gcloud_GeoTIFF_Nov2022.ipynb` contains Google Earth Engine Python API code to export Landsat satellite imagery (geoTIFF format) from Google Earth Engine. Imagery is exported as mosaics sampled from three-year sequences into a Google Cloud bucket.
    - `Model_Training_Jan2023_SpatialEncoding.ipynb` contains Tensorflow/Keras functions to train a Temporal CNN deep learning model on Landsat spectral band information. A trained model can predict leafy spurge from Landsat imagery. Included are several functions to spatial thin leafy spurge occurrence points, calculate model performance metrics, and confusion matrices. Different model architectures were investigated in similar notebook versions (Sept2022, Oct2022, Nov2022.ipynb.. ). 
    - `Landsat_Gap_Fill_Mean_Pixel_Mosaic.ipynb` contains code to gap-fill errors that arose when mosaicing Landsat 5 imagery from before the year 2000's. Absent imagery during the mosaic process created 0-data values in the Landsat images that can cause issues when running model predictions. The Python scripts in the notebook include routines to replace those 0-data values with mean sampled data from other Landsat spectral bands, essentially replacing missing data.
    - `Prediction_Timeseries_LinearRegression.ipynb` contains Python code to perform linear regressions on predictions of leafy spurge across 35 years. The code fits a Ricker model to time-series data on the probability of leafy spurge in time, with each time-step represented by a .tif raster. 

- The `/datasets_oct22` directory contains .CSV files with Landsat satellite spectral data sampled from labeled points, as used in the TemporalCNN models and sampled from the files: `TemporalCNN_GenerateTrainingData_Points_Gcloud.ipynb` or `/pythonFiles/TemporalCNN_GenerateTFRecordPoints.py`. 
    - The master dataset: `'NLCD_full_dataset_allyears_latlong_thinned_001dd_spurge_apr2023.csv'` contains several million rows of labeled point data and Landsat spectral information. Thinned_001dd indicates the leafy spurge points were thinned to 0.001 decimal degrees to reduce spatial clustering/bias in the dataset. 

- The `/Demographic_Ricker_Models` directory contains .tif mosaics of the growth rate and carrying capacity values fit from linear models on pixel-wise leafy spurge predictions. Mosaics output from `/analysesNotebooks/Prediction_Timeseries_LinearRegression.ipynb` or `/pythonFiles/predictRasters-densetimesteps-correctedLandsat-may23.py`. Two folders contain predictions from various Landsat imagery that was uncorrected (not gap-filled of 0-data, May 2023) and corrected (gap-filled of 0-data, June 2023). 

- The `/Env_Variables` directory contains environmental variables as .tif raster data. Climate, elevation, slope, and topographic position index are included for analyses. Species distribution model predictions from Lake et al., 2019, are also included.

- The `/model_predictions` directory contains .tif mosaics of the predicted probability of leafy spurge from 12 epochs spanning 35 years (1985 - 2019). Two folders contain predictions from various Landsat imagery that was uncorrected (not gap-filled of 0-data, May 2023) and corrected (gap-filled of 0-data, June 2023). 

- The `/pythonFiles` directory contains Python scripts for Slurm submission. See `slurmScripts` directory for submission scripts of individual Python files. See `/analysesNotebooks` for development version and more verbose code documentation for each file. Files starting with `predictRasters-*.py` are various stages of model predictions that iterate across folders containing Landsat satellite imagery to predict leafy spurge and other land cover classes based spectral data in Landsat pixels. 

- The `/slurmScripts` directory contains scripts to submit `/pythonFiles` into the Slurm scheduler. For certain files (e.g., `predictRasters-*.py`), Slurm arrays are used to batch predictions for each Landsat tile.

- The `/temporalCNN` directory contains trained models and evaluation metrics for individual models. Important models are `/Archi3` (top model) with the highest performance. However, this model is likely overfit on leafy spurge sampled that were not spatially thinned to 0.001 decimal degrees. `/Archi6` (thin model) has intermediate performance and was trained on spatially-thinned leafy spurge occurrences (to 0.001 decimal degrees).

Other directories: `jsonKeys` contains a private .json file with access to a Google Cloud service account hosting data, models, and predictions. Do not distribute.

Other files: [model_predictions_workflow.txt](https://github.com/lake-thomas/landsat-leafy-spurge/blob/main/model_predictions_workflow.txt) contains routines for moving, manipulating, and mosaicing various types of training data and model predictions. This collection of scripts is useful as a reference when, for example, a user wants to move a .tif image from MSI to Google Cloud to Google Earth Engine.

Other files: [spurge_block_crossvalidation_point_thinning_landsat_april2023.Rmd](https://github.com/lake-thomas/landsat-leafy-spurge/blob/main/spurge_block_crossvalidation_point_thinning_landsat_april2023.Rmd) contains R code for dividing the master `datasets_oct22/NLCD_full_dataset_allyears_latlong_thinned_001dd_spurge_apr2023.csv` data file into spatially cross-validated training, testing, and validation sets based on 1 decimal degree latitude/longitude partitions. A "checkerboard" pattern results in training, testing, and validation datasets that were used during model training.
