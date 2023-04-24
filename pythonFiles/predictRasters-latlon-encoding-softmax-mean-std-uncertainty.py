#!/usr/bin/python

# April 2023 Thomas Lake
# Partially-automates the model prediction process.
# For each Landsat tile (0-258), first get spatial point data (lat/lon) of every tile raster cell.
# For each of the 10 land cover classes, predict the model 5 times to estimate uncertainty.
# For each of the 5 predictions/class, calculate the mean and standard deviation of predicted probability in each raster cell.
# Convert the model predictions into the Unsigned Interger 16 format to save disk space. Predictions scaled from float64 as 0-1 to uint16 as 0-65535.
# Write two files: mean prediction for each class and standard deviation in prediction for each class.

#Load packages

# Packages
import os
import pandas as pd
import numpy as np
import datetime
import pprint
import time
import math
import random
import glob
from functools import reduce
from pprint import pprint
from itertools import tee

# Plotting
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Tensorflow version 2.4.1
import tensorflow as tf
print(tf.__version__) 

# Keras setup.
import keras
from keras import layers
from keras.layers import Flatten
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras.regularizers import l2
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Lambda, SpatialDropout1D, Concatenate
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import Callback, ModelCheckpoint, History, EarlyStopping
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras import backend as K

#Prediction
import rasterio as rio
rio.env.local._CHECK_DISK_FREE_SPACE = False
import glob
import time

# Specify Arguments for file input (which raster to predict
from sys import argv

# Argument (interger) specifies what raster to predict in a list of rasters
input_value = int(argv[1])

# Load a trained model
model = keras.models.load_model(r'/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/temporalCNN/Archi3/TemporalCNN_100epochs_latlongenc_dropout_uncertainty_topmodel_apr122023.h5')

# Input prediction .tif path
image_path = r'/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/landsat_tifs_2019/'

# Output prediction file path
outpath = r'/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/model_predictions/predictions_2019/'

# List all .tif files in /rasters folder for prediction
tif_image_list = glob.glob(image_path + '*.tif')

print(tif_image_list[input_value])

#record how long a prediction takes
prediction_train_time = time.time()

with rio.open(tif_image_list[input_value], 'r') as dataset:
    # First, get the coordinates of every pixel in the .tif image
    # Define shape of .tif image
    shape = dataset.shape
    nodata = dataset.nodata

    coords_get_time = time.time()

    #Get the X,Y coordinates (lat/long) for each dataset (image) and index as np array
    xy1, xy2 = tee(dataset.xy(x, y) for x, y in np.ndindex(shape))  # save re-running dataset.xy
    data = ((x, y, z[0]) for (x, y), z in zip(xy1, dataset.sample(xy2)) if z[0] != nodata)
    res = pd.DataFrame(data, columns=["lon", "lat", "data"])
    coords = res.to_numpy() #convert to numpy array
    coords2 = coords[:,0:2] # Remove 'data' column, make latitude come before longitude
    coords2[:,[1,0]] = coords2[:,[0,1]] # swap longitude and latitude columns
   
    print("Got Coordinates of Landsat Image \n")
    coords_time = round(time.time()-coords_get_time, 2)
    print("Seconds to Calculate Coordinates:", coords_time)


    # Second, get the spectral data from every pixel in the .tif image
    arr = dataset.read()
    # Define shape of input .tif image
    bands, width, height = arr.shape

    # Convert Tif Data Type to float32 by division.
    arr = arr/10000

    # Reshape .tif array axes for correct format so model can predict.
    arr = np.moveaxis(arr, 0, -1) #move axis to channels last
    new_arr = arr.reshape(-1, arr.shape[-1]) #reshape to row and column
    num_pixels = width*height
    spectral = new_arr.reshape(num_pixels, 9, 7)
    #print(spectral.shape)

    #combine both latitude/longitude and spectral data into list for model prediction
    X_pred = [coords2, spectral]
    print("Got Spectral Data\n")

    predictions_get_time = time.time()

    #Prediction 1
    p_1 = model.predict(X_pred) # p is prediction from the model, yields (nrow * ncol, 10 classes)
    pim_1 = p_1.reshape(width, height, 10) # Dimension of prediction is now (nrow, ncol, 10 classes)
    pim_1 = np.moveaxis(pim_1, 2, 0) # move axis so classes/bands is first, dimensions now (10, nrow, ncol)

    print("Prediction 1 Completed")
    prediction_time = round(time.time()-predictions_get_time, 2)
    print("Seconds to Prediction 1:", prediction_time)

    #print(pim_1.shape)

    #Prediction 2
    p_2 = model.predict(X_pred) # p is prediction from the model, yields (nrow * ncol, 10 classes)
    pim_2 = p_2.reshape(width, height, 10) # Dimension of prediction is now (nrow, ncol, 10 classes)
    pim_2 = np.moveaxis(pim_2, 2, 0) # move axis so classes/bands is first, dimensions now (10, nrow, ncol)

    print("Prediction 2 Completed")
    prediction_time = round(time.time()-predictions_get_time, 2)
    print("Seconds to Prediction 2:", prediction_time)        

    #print(pim_2.shape)

    #Prediction 3
    p_3 = model.predict(X_pred) # p is prediction from the model, yields (nrow * ncol, 10 classes)
    pim_3 = p_3.reshape(width, height, 10) # Dimension of prediction is now (nrow, ncol, 10 classes)
    pim_3 = np.moveaxis(pim_3, 2, 0) # move axis so classes/bands is first, dimensions now (10, nrow, ncol)

    print("Prediction 3 Completed")
    prediction_time = round(time.time()-predictions_get_time, 2)
    print("Seconds to Prediction 3:", prediction_time)

    #print(pim_3.shape)

    #Prediction 4
    p_4 = model.predict(X_pred) # p is prediction from the model, yields (nrow * ncol, 10 classes)
    pim_4 = p_4.reshape(width, height, 10) # Dimension of prediction is now (nrow, ncol, 10 classes)
    pim_4 = np.moveaxis(pim_4, 2, 0) # move axis so classes/bands is first, dimensions now (10, nrow, ncol)

    print("Prediction 4 Completed")
    prediction_time = round(time.time()-predictions_get_time, 2)
    print("Seconds to Prediction 4:", prediction_time)

    #print(pim_4.shape)

    #Prediction 5
    p_5 = model.predict(X_pred) # p is prediction from the model, yields (nrow * ncol, 10 classes)
    pim_5 = p_5.reshape(width, height, 10) # Dimension of prediction is now (nrow, ncol, 10 classes)
    pim_5 = np.moveaxis(pim_5, 2, 0) # move axis so classes/bands is first, dimensions now (10, nrow, ncol)

    print("Prediction 5 Completed")
    prediction_time = round(time.time()-predictions_get_time, 2)
    print("Seconds to Prediction 5:", prediction_time)

    #print(pim_5.shape)

    # Calculate mean and standard deviation for each class
    class_means = np.mean(np.array([pim_1, pim_2, pim_3, pim_4, pim_5]), axis=0) # average predicted value across class axis 0
    class_stds = np.std(np.array([pim_1, pim_2, pim_3, pim_4, pim_5]), axis=0) # standard deviation of predicted value across class axis 0
    class_cvs = np.array(class_stds / class_means) # coefficient of variation of predicted value across class axis 

    # Get the file name (landsat_image_170_t.tif) by splitting input path.
    fileout_string = os.path.split(tif_image_list[input_value])

    # Output prediction raster .
    out_meta = dataset.meta.copy()

    # Get Output metadata.
    out_meta.update({'driver':'GTiff',
                     'width':dataset.shape[1],
                     'height':dataset.shape[0],
                     'count':class_means.shape[0],
                     'dtype':'uint16',
                     'crs':dataset.crs, 
                     'transform':dataset.transform})

    
    # Scale the class_means array to integers between 0 and 65535
    scaled_class_means = (class_means * 65535).astype(np.uint16)
    
    # Write the mean array to a multiband raster file
    with rio.Env(CHECK_DISK_FREE_SPACE=False):
        with rio.open(fp=outpath + "/mean_prediction_softmax_classes_uint16_" + fileout_string[-1], mode='w',**out_meta) as dst:
            dst.write(scaled_class_means)
            

    # Scale the class_cvs array to integers between 0 and 65535
    # To scale the integer values back to the 0-1 scale, you can simply divide the integer values by 65535.0 to get floating-point values between 0 and 1. float_data = int_data / 65535.0
    scaled_class_cvs = (class_cvs * 65535).astype(np.uint16)
    
    # Write the stdev array to a multiband raster file
    with rio.Env(CHECK_DISK_FREE_SPACE=False):
        with rio.open(fp=outpath + "/coefvar_prediction_softmax_classes_uint16_" + fileout_string[-1], mode='w',**out_meta) as dst:
            dst.write(scaled_class_cvs)

    print("Writing files... \n")
    prediction_time = round(time.time()-prediction_train_time, 2)
    print("Total Time: ", prediction_time)        
    
    
    
#EOF