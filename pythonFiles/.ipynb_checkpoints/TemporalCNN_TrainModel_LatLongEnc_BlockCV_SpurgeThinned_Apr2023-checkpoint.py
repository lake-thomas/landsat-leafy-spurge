#!/usr/bin/python

#Script uses conda environment 'tf_gpu_earthengine' from conda env 'tf_gpu'
#Feb 24 2023
#Author: Thomas Lake

# Specify Arguments for file input (to select bounding box and extract training datasets)
import sys
from sys import argv

# Argument (interger) specifies model run number
run_number = int(argv[1])


# Packages
import os
import pandas as pd
import numpy as np
import datetime
import pprint
import time
import math
import random
from glob import glob
from functools import reduce
from pprint import pprint
import itertools

# Plotting
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Tensorflow setup

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


# Import from ~/sits folder
# Contains readingsits.py file to read and compute and reshape the SITS data
sys.path.append("/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/temporalCNN/sits")
import readingsits

# Import from ~/deeplearning folder
# Contains multiple .py files with varying DL architectures 
sys.path.append("/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/temporalCNN/deeplearning")

import architecture_features
import architecture_complexity
import architecture_rnn
import architecture_regul
import architecture_batchsize
import architecture_depth
import architecture_spectro_temporal
import architecture_pooling

# Import from ~/outputfiles folder
# Contains evaluation.py and save.py files with fucntions to compute summary statistics, write predictions, and create confusion matrices
sys.path.append("/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/temporalCNN/outputfiles")

import evaluation
import save

# Set the path to exported training/testing dataset
sits_path = '/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/datasets_oct22'    
    
    
#-----------------------------------------------------------------------
# Parameters to set
n_channels = 7 #-- B G NDVI NIR Red SWIR1 SWIR2
val_rate = 0.1 # Validation data rate

# String variables for the training and testing datasets
train_str = 'NLCD_train_dataset_allyears_latlong_thinned_001dd_spurge_apr2023'
test_str = 'NLCD_test_dataset_allyears_latlong_thinned_001dd_spurge_apr2023'					

# Get filenames
train_file = sits_path + '/' + train_str + '.csv'
test_file = sits_path + '/' + test_str + '.csv'
print("train_file: ", train_file)
print("test_file: ", test_file)

# Set a model results path
res_path = '/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/temporalCNN'

# Set Architecture / Model Run Index (used if running in batch on MSI)
noarchi = 7
norun = run_number
feature = "SB" #use only spectral bands provided (do not compute new bands, like NDVI, which are already computed)
train_str = "latlong-spurgespatialthin001-blockcrossvalidation"

# Creating output path if does not exist
if not os.path.exists(res_path):
  print("ResPath DNE")
  os.makedirs(res_path)

# Output files			
res_path = res_path + '/Archi' + str(noarchi) + '/'
if not os.path.exists(res_path):
  os.makedirs(res_path)
  print("noarchi: ", noarchi)

# Create output files to capture model results
str_result = feature + '-' + train_str + '-noarchi' + str(noarchi) + '-norun-' + str(norun)

# Output for model evaluation metrics
res_file = res_path + 'result_accuracy_metrics-' + str_result + '.txt'

# Output for model loss / epochs
traintest_loss_file = res_path + 'trainingHistory-' + str_result + '.txt'

# Output for confusion matrix
conf_file = res_path + 'confMatrix-' + str_result + '.txt'

# Output for model weights file (.h5 file)
out_model_file = res_path + 'model-' + str_result + '.h5'

print("Model accuracy metrics: " + res_file)
print("Model history: " + traintest_loss_file)
print("Model confusion matrix: " + conf_file)
print("Model output file: " + out_model_file)



###
# Load Datasets
###


#Import training, testing, validation datasets
from tensorflow.keras.utils import to_categorical

# Read training set dataframe
train_df = pd.DataFrame(pd.read_csv("/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/datasets_oct22/NLCD_train_dataset_allyears_latlong_thinned_001dd_spurge_apr2023.csv", header=1))
#train_df.head()
#print(train_df.shape)
nchannels = 7 #-- B G NDVI NIR Red SWIR1 SWIR2

# Select spectral data (first 63 columns)
batch_X = train_df.iloc[:, 1:64].to_numpy()
batch_X_spectral = batch_X.reshape(batch_X.shape[0],int(batch_X.shape[1]/nchannels),nchannels)
#print(batch_X_spectral[0])
#print(batch_X_spectral.shape)

# Select latitude/longitude coordinates (get last 3 columns, remove last 1 (class) column)
batch_X_coords = train_df.iloc[:, -3:].iloc[:, :-1].to_numpy()
#print(batch_X_coords.shape)

# Add X data(lat/long coordinates and spectral data) as list to X_train
# Add Y data(class) to Y_train
X_train = [batch_X_coords, batch_X_spectral]

# Select class data from last column
y_train = np.asarray(train_df.iloc[:, -1]).astype(int)
#print(batch_Y.shape)
#print(batch_Y)
y_train_one_hot = to_categorical(y_train)
#print(Y_one_hot.shape)

print(X_train[0].shape, X_train[1].shape, y_train_one_hot.shape)





# Read testing set dataframe
test_df = pd.DataFrame(pd.read_csv("/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/datasets_oct22/NLCD_test_dataset_allyears_latlong_thinned_001dd_spurge_apr2023.csv", header=1))
#test_df.head()
#print(test_df.shape)
nchannels = 7 #-- B G NDVI NIR Red SWIR1 SWIR2

# Select spectral data (first 63 columns)
batch_X = test_df.iloc[:, 1:64].to_numpy()
batch_X_spectral = batch_X.reshape(batch_X.shape[0],int(batch_X.shape[1]/nchannels),nchannels)
#print(batch_X_spectral[0])
#print(batch_X_spectral.shape)

# Select latitude/longitude coordinates (get last 3 columns, remove last 1 (class) column)
batch_X_coords = test_df.iloc[:, -3:].iloc[:, :-1].to_numpy()
#print(batch_X_coords.shape)

# Add X data(lat/long coordinates and spectral data) as list to X_test
# Add Y data(class) to Y_test
X_test = [batch_X_coords, batch_X_spectral]

# Select class data from last column
y_test = np.asarray(test_df.iloc[:, -1]).astype(int)
#print(batch_Y.shape)
#print(batch_Y)
y_test_one_hot = to_categorical(y_test)
#print(Y_one_hot.shape)

print(X_test[0].shape, X_test[1].shape, y_test_one_hot.shape)





# Read validation set dataframe
valid_df = pd.DataFrame(pd.read_csv("/panfs/jay/groups/31/moeller/shared/leafy-spurge-demography/datasets_oct22/NLCD_valid_dataset_allyears_latlong_thinned_001dd_spurge_apr2023.csv", header=1))
#valid_df.head()
#print(valid_df.shape)
nchannels = 7 #-- B G NDVI NIR Red SWIR1 SWIR2

# Select spectral data (first 63 columns)
batch_X = valid_df.iloc[:, 1:64].to_numpy()
batch_X_spectral = batch_X.reshape(batch_X.shape[0],int(batch_X.shape[1]/nchannels),nchannels)
#print(batch_X_spectral[0])
#print(batch_X_spectral.shape)

# Select latitude/longitude coordinates (get last 3 columns, remove last 1 (class) column)
batch_X_coords = valid_df.iloc[:, -3:].iloc[:, :-1].to_numpy()
#print(batch_X_coords.shape)

# Add X data(lat/long coordinates and spectral data) as list to X_valid
# Add Y data(class) to Y_valid
X_valid = [batch_X_coords, batch_X_spectral]

# Select class data from last column
y_valid = np.asarray(valid_df.iloc[:, -1]).astype(int)
#print(batch_Y.shape)
#print(batch_Y)
y_valid_one_hot = to_categorical(y_valid)
#print(Y_one_hot.shape)

print(X_valid[0].shape, X_valid[1].shape, y_valid_one_hot.shape)




###
# Build the Model in Keras
###

# Build the Temporal CNN
# Model Part 1 encodes the latitude/longitude information
# Model Part 2 encodes the spectral band information
# At the end, we concatenate (merge) models 
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input

#Get model input sizes
input_shape = (9, 7) #spectral data dimensions (batch size, timesteps, channels)
input_coords_shape = (2,) #lat long coords (batch size, [lat, long])

# Model 1
    # Latitude/Longitude Spatial Encoding
    # Define Input for Spatial Data
X_coords_input = Input(shape=(input_coords_shape))
    #-- lat long coordinates fc + relu
model1_out = Dense(576, activation='relu')(X_coords_input) #Dense 576 = 9 (timesteps) * 64 (conv1d filters)
model1 = Model(inputs = [X_coords_input],  outputs = [model1_out])

# Model 2
    # Temporal Convolutional Neural Network (1D-CNN)
#-- parameters of the architecture
nbclasses = 10 # number of classes for output
l2_rate = 1.e-6 # regularization
dropout_rate = 0.1 # regularization
nbunits_conv = 64 #-- convolution filters, will be double
nbunits_fc = 128 #-- fully connected (dense) layer, will be double    
    
	# Define the input placeholders.
X_input = Input(input_shape) # (batch size, timesteps, channels)
    #-- Conv BN Activation Dropout
X = Conv1D(filters = 64, kernel_size = 5, strides = 1, dilation_rate=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_rate))(X_input)
X = BatchNormalization(axis=-1)(X)
X = Activation('relu')(X)
X = Dropout(dropout_rate)(X) 
X = Conv1D(filters = 64, kernel_size = 5, strides = 1, dilation_rate=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_rate))(X)
X = BatchNormalization(axis=-1)(X)
X = Activation('relu')(X)
X = Dropout(dropout_rate)(X)
X = Conv1D(filters = 128, kernel_size = 5, strides = 1, dilation_rate=4, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(l2_rate))(X)
X = BatchNormalization(axis=-1)(X)
X = Activation('relu')(X)
X = Dropout(dropout_rate)(X)
model2_out = Flatten()(X)
model2 = Model(inputs = [X_input], outputs = [model2_out])

# Combine two models on common axis 576
concatenated = Concatenate(axis=-1)([model1_out, model2_out])

# Model 3 (combined model)
    # Fully connected layer and softmax output
Z = Dense(nbunits_fc, kernel_initializer="he_normal", kernel_regularizer=l2(l2_rate))(concatenated)
Z = BatchNormalization(axis=-1)(Z)
Z = Activation('relu')(Z)
Z = Dropout(dropout_rate)(Z, training=True) #adding training = True to dropout allows us to use dropout during training and inference (for uncertainty)
	#-- SOFTMAX layer
out = Dense(nbclasses, activation='softmax', kernel_initializer="he_normal",kernel_regularizer=l2(l2_rate))(Z)

# Create merged model.
merged_model = Model(inputs = [X_coords_input, X_input], outputs = out)






###
# Define Model Variables and Compile Model
###
#Define Class Weights
from sklearn.utils import class_weight

class_weights =  class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)

class_weights = dict(zip(np.unique(y_train), class_weights))

# Class weights function
# inverse of frequency?
class_weights = {0: 0,
                 1: 1.046028630719989,
                 2: 1.6421837069230087,
                 3: 1.37461158722999,
                 4: 0.7614511317372198,
                 5: 0.6015453322153169,
                 6: 0.3652990948014909,
                 7: 0.39487324200412083,
                 8: 4.334510403657227,
                 9: 26}

print(class_weights)


# Set Model variables
n_epochs = 300
batch_size = 10192
lr = 0.0001 # learning rate for optimizer
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-07

#Model Optimizer (adam)
opt = tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

# Set Loss function 
cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Compile Model
merged_model.compile(optimizer = opt, loss = cce, metrics = ["accuracy"])

merged_model.summary()





###
# Define model callbacks
###

# Save Model at Regular Checkpoints
checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1, mode='auto', restore_best_weights=False)

#Plot Loss and Accuracy Callback
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.f1 = []
        self.val_f1 = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.f1.append(logs.get('accuracy'))
        self.val_f1.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val loss")
        ax1.legend()
        
        ax2.plot(self.x, self.f1, label="Acc")
        ax2.plot(self.x, self.val_f1, label="val Acc ")
        ax2.legend()
        
        plt.show();
        
plot_losses = PlotLearning()

class LROutput(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(lr)
        
lr_output = LROutput()



###
# Define Learning Rate as Cosine Decay with Warmup
### 


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


   
# Number of training samples (rows).
sample_count = X_train[0].shape[0]

# Total epochs to train.
epochs = n_epochs

# Number of warmup epochs. (10% of total epochs)
warmup_epoch = 30

# Training batch size, set small value here for demonstration purpose.
batch_size = batch_size

# Base learning rate after warmup.
learning_rate_base = 0.001

total_steps = int(epochs * sample_count / batch_size) #98,275

# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size) #9827

# Compute the number of warmup batches.
warmup_batches = warmup_epoch * sample_count / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)






###
# Model Training
###


#Train the model
start_train_time = time.time()

# Fit the model
hist = merged_model.fit(x = X_train, 
                 y = y_train_one_hot, 
                 epochs = n_epochs,
                 batch_size = batch_size,
                 validation_data = (X_valid, y_valid_one_hot),
                 shuffle=True,
                 verbose=1,
                 class_weight = class_weights,
                 callbacks = [warm_up_lr])


train_time = round(time.time()-start_train_time, 2)



# Save the Trained Model as a .h5 file
merged_model.save(out_model_file)
           




###
# Evaluate the model prediction
###

from sklearn.metrics import multilabel_confusion_matrix
from tabulate import tabulate

# Predict the model on withheld testing dataset
y_pred = merged_model.predict(X_test)

y_pred = np.argmax(y_pred, axis=-1)
y_pred_flat = y_pred.flatten()
y_pred_flat = y_pred_flat.astype(int)

y_test = y_test.astype(int)    
y_test_flat = y_test.flatten()

#Count number of classes in the testing dataset
unique_vals, counts = np.unique(y_test, return_counts=True)

# Calculate confusion matrix
class_names = ["Water", "Developed", "BarrenLand", "Forest", "Shrub/Scrub", "Grassland/Herbaceous", "Croplands", "EmergentWetlands", "LeafySpurge"]
class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
c = multilabel_confusion_matrix(y_test_flat, y_pred_flat, labels = class_labels)
model_output_metrics = []
for i in range(len(class_labels)):
    tn=c[i, 0, 0]
    tp=c[i, 1, 1]
    fn=c[i, 1, 0]
    fp=c[i, 0, 1]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    TPR_Sens_Recall = tp/(tp+fn)
    TNR_Spec = tn/(tn+fp)
    FPR = fp/(fp+tn)
    FNR = fn/(fn+tp)
    precision = tp/(tp+fp)
    jaccard = tp/(tp+fp+fn)
    beta = 0.5
    F05 = ((1 + beta**2) * precision * TPR_Sens_Recall) / (beta**2 * precision + TPR_Sens_Recall)
    beta = 1
    F1 = ((1 + beta**2) * precision * TPR_Sens_Recall) / (beta**2 * precision + TPR_Sens_Recall)
    beta = 2
    F2 = ((1 + beta**2) * precision * TPR_Sens_Recall) / (beta**2 * precision + TPR_Sens_Recall)
    outputs = [class_names[i], counts[i], tp, tn, fp, fn, accuracy, TPR_Sens_Recall, TNR_Spec, FPR, FNR, precision, jaccard, F2]
    model_output_metrics.append(outputs)

# Print and format outputs
print(tabulate(model_output_metrics, floatfmt=".2f", headers=["Class Name", "Num Points", "TP", "TN", "FP", "FN", "Accuracy", "TPR/Sens/Recall", "TNR/Spec", "FPR", "FNR", "Precision", "Jaccard", "F2"]))

#Save model results to file
with open(res_file, 'w') as f:
    f.write(tabulate(model_output_metrics, floatfmt=".2f", headers=["Class Name", "TP", "TN", "FP", "FN", "Accuracy", "TPR/Sens/Recall", "TNR/Spec", "FPR", "FNR", "Precision", "Jaccard", "F1"]))
    

# Save losses and accuracy
train_loss = hist.history['loss']
val_loss   = hist.history['val_loss']
train_acc  = hist.history['accuracy']
val_acc    = hist.history['val_accuracy']
xc         = range(n_epochs)

traintest_loss_df = pd.DataFrame(
    {'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'epochs': xc
    })

traintest_loss_df
    
np.savetxt(traintest_loss_file, traintest_loss_df, fmt='%6f')



import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread, imshow, imsave
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, fbeta_score
from tensorflow.keras.models import load_model
from tabulate import tabulate

def plot_confusion_matrix(
        y_true,
        y_pred,
        classes,
        test_name,
        normalize=False,
        set_title=False,
        save_fig=False,
        cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if set_title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # and save it to log file
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        #with open(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_log.txt', 'ab') as f:
        #    f.write(b'\nNormalized confusion matrix\n')
        #    np.savetxt(f, cm, fmt='%.3f')
    else:
        print('Confusion matrix, without normalization')
        #with open(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_log.txt', 'ab') as f:
        #    f.write(b'\nConfusion matrix, without normalization\n')
        #    np.savetxt(f, cm, fmt='%7u')

    #print(cm)
    #cm = cm[1:10]
    #cm = cm[:,1:]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if normalize:
        im.set_clim(0., 1.)     # fixes missing '1.0' tick at top of colorbar
    cb = ax.figure.colorbar(im, ax=ax)
    if normalize:
        cb.set_ticks(np.arange(0., 1.2, 0.2))
        cb.set_ticklabels([f'{i/5:.1f}' for i in range(6)])
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title if set_title else None,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(len(cm)-0.5, -0.5)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if np.round(cm[i, j], 2) > 0.:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, '–',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if save_fig:
        if normalize:
            plt.savefig(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_cm_normal.pdf')
        else:
            plt.savefig(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_cm_non_normal.pdf')
    return fig, ax, cm




inv_category_dict = {1:"Water", 2:"Developed", 3:"BarrenLand", 4:"Forest", 5:"Shrub/Scrub", 6:"Grassland/Herbaceous", 7:"Croplands", 8:"EmergentWetlands", 9:"LeafySpurge"}

class_names = [inv_category_dict[i] for i in np.arange(1, 10)]


# save plot of normalized cm
cm = plot_confusion_matrix(
    y_test_flat,
    y_pred_flat,
    classes=class_names,
    test_name="myModel",
    normalize=True,
    save_fig=False
)

conf_df = pd.DataFrame(cm[2], columns = ["Water", "Developed", "BarrenLand", "Forest", "Shrub/Scrub", "Grassland/Herbaceous", "Croplands", "EmergentWetlands", "LeafySpurge"])

np.savetxt(conf_file, conf_df, fmt='%6f')


#EOF