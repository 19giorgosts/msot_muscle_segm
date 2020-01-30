#from google.colab import drive
#drive.mount('/content/gdrive')
import warnings
warnings.filterwarnings("ignore")

#target_height = 400
#target_width = 400

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import os
import h5py
import time
import keras
import cv2 # cv2.imread() for grayscale images
%matplotlib inline
matplotlib.use('Agg')
import sys
import skimage.transform
import random as r
from keras.preprocessing import image# for RGB images
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import AveragePooling2D,GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from keras.layers.merge import concatenate
#from mpl_toolkits import axes_grid1
from tqdm import tqdm
from numpy import ndarray as nd
from sklearn.model_selection import train_test_split
#######################################


#path='/home/giorgostsekas92/msot_muscle_segmentation/MSOT_data/'
#%cd MSOT_data/
#data='msot_data_baseline'
#Nimages = len(os.listdir(path+data)) # total images in path
## config file reading and data loading

CURRENT_PATH = os.getcwd()
user_config = configparser.RawConfigParser()
user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)
data_load(options)

run_random(5,5,options['data_folder']+'X_ts.npy'.shape[0]) # run_random(Nmodels,Ndata,Ndatapoints)
run_sug(5,5,options['data_folder']+'X_ts.npy'.shape[0]) # run_sg(Nmodels,Ndata,Ndatapoints)


import json
with open('dice_msot_rand.json') as json_file:
    dice_rand = json.load(json_file)
with open('dice_msot_sg.json') as json_file:
    dice_sg = json.load(json_file)

##plotting results
X_rand,X_sg = parsing(options['Nmodels'],options['Ndata'])
generate_plot(X_rand,X_sg,options['Ndata'])



