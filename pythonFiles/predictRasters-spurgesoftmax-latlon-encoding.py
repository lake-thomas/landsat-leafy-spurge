#!/usr/bin/python

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
from rasterio.plot import show
import glob
import time

# Specify Arguments for file input (which raster to predict
from sys import argv

# Argument (interger) specifies what raster to predict in a list of rasters
input_value = int(argv[1])

# Load a trained model
model = keras.models.load_model(r'/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/temporalCNN/Archi3/TemporalCNN_100epochs__latlongenc_dropout_uncertainty_apr42023.h5')

# Input prediction .tif path
image_path = r'/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/landsat_tifs_2019/'

# Output prediction file path
outpath = r'/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/model_predictions/predictions_2019/iteration_4/'

# List all .tif files in /rasters folder for prediction
tif_image_list = glob.glob(image_path + '*.tif')

print(tif_image_list[input_value])

#record how long a prediction takes
prediction_train_time = time.time()

# Open .tif array image with rasterio, read to numpy array
with rio.open(tif_image_list[input_value], 'r') as dataset:
    # First, get the coordinates of every pixel in the .tif image
    # Define shape of .tif image
    shape = dataset.shape
    nodata = dataset.nodata

    xy1, xy2 = tee(dataset.xy(x, y) for x, y in np.ndindex(shape))  # save re-running dataset.xy
    data = ((x, y, z[0]) for (x, y), z in zip(xy1, dataset.sample(xy2)) if z[0] != nodata)
    res = pd.DataFrame(data, columns=["lon", "lat", "data"])
    coords = res.to_numpy() #convert to numpy array
    coords2 = coords[:,0:2] # Remove 'data' column, make latitude come before longitude
    coords2[:,[1,0]] = coords2[:,[0,1]] # swap longitude and latitude columns
    #print(coords2[1:10, :], coords2.shape)
    print("Got Coordinates of Landsat Image \n")

    # Second, get the spectral data from every pixel in the .tif image
    arr = dataset.read() #read .tif as array
    # Define shape of input .tif image
    bands, width, height = arr.shape

    # Convert Tif Data Type to float32 by division.
    arr = arr/10000

    # Reshape .tif array axes for correct format so model can predict.
    arr = np.moveaxis(arr, 0, -1) #move axis to channels last
    new_arr = arr.reshape(-1, arr.shape[-1]) #reshape to row and column
    num_pixels = width*height
    spectral = new_arr.reshape(num_pixels, 9, 7)
    print(spectral.shape)

    #combine both latitude/longitude and spectral data into list for model prediction
    X_pred = [coords2, spectral]
    print("Got Spectral Data \n")

    # Predict model and reshape to export.
    p = model.predict(X_pred) # p is prediction from the DL model
    pim = p.reshape(width, height, 10) # Dimension of prediction in rows, columns, bands (10 classes)

    # Get the file name (landsat_image_170_t.tif) by splitting input path.
    fileout_string = os.path.split(tif_image_list[input_value])

    # Output prediction raster .
    out_meta = dataset.meta.copy()

    # Get Output metadata.
    out_meta.update({'driver':'GTiff',
                     'width':dataset.shape[1],
                     'height':dataset.shape[0],
                     'count':1,
                     'dtype':'float64',
                     'crs':dataset.crs, 
                     'transform':dataset.transform,
                     'nodata':0})

    # Write predicted raster to file.
    with rio.open(fp=outpath + "/prediction_" + fileout_string[-1], #outputpath_name
                 mode='w',**out_meta) as dst:
                 dst.write(pim[:, :, 9], 1) # output only leafy spurge softmax

    print("Writing file... \n")
    prediction_time = round(time.time()-prediction_train_time, 2)
    print(prediction_time)
    

#EOF