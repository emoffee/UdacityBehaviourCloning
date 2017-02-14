
# coding: utf-8

## Initializations of packages/parameters/path
import os,cv2,json,time,csv
from pathlib import Path
import numpy as np
from numpy.random import random
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ELU ,Lambda 
from keras.utils import np_utils
from keras.optimizers import Adam

DrivingLog = './driving_log.csv'
OutputModelJson = 'model.json'
OutputModelH5 = 'model.h5'

batch_size = 20
nb_classes = 1
nb_epoch = 500
channel,rows, cols = 3, 160, 320 
input_shape=(channel,rows,cols)
pool_size = (2, 3)


# Pre-defined Functions
# Crop the image by its top one third
def locrop_image(imgpath):
    imagepath = imgpath.replace(' ','')
    image = cv2.imread(imagepath, 1)
    shape = image.shape
    image = image[int(shape[0]/3):shape[0], 0:shape[1]]
    image = cv2.resize(image, (rows, cols), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

# Image generator.
def image_generator(X, Y, rows, cols, channel):  
    while 1:
        for i in range(len(X)):
            y = Y[i]
            if y < -0.01:
                rdm = random()
                if rdm > 0.75:
                    imgpath = X[i].split('?')[1]
                    y *= 3.0
                else:
                    if rdm > 0.5:
                        imgpath = X[i].split('?')[1]
                        y *= 2.0
                    else:
                        if rdm > 0.25:
                            imgpath = X[i].split('?')[0]
                            y *= 1.5
                        else:
                            imgpath = X[i].split('?')[0]
            else:
                if y > 0.01:
                    rdm = random()
                    if rdm > 0.75:
                        imgpath = X[i].split('?')[2]
                        y *= 3.0
                    else:
                        if rdm > 0.5:
                            imgpath = X[i].split('?')[2]
                            y *= 2.0
                        else:
                            if rdm > 0.25:
                                imgpath = X[i].split('?')[0]
                                y *= 1.5
                            else:
                                imgpath = X[i].split('?')[0]
                else:
                    imgpath = X[i].split('?')[0]
            
            ximage = locrop_image(imgpath)
            y_train = np.array([[y]])
            x_train = ximage.reshape(1,channel, rows, cols)
            yield x_train, y_train
            
## Pre-processing Data
## Before we start to process the data, we would like to analyze the driving_csv log
## features: center,left,right,steering,throttle,brake,speed

Col_Names=['Center Image','Left Image','Right Image','Steering Angle','Throttle','Break','Speed']
driving_csv = pd.read_csv(DrivingLog,header=None,names=Col_Names)
print("1. Number of Driving Records: %d" % len(driving_csv))
print("---------------------------------------")
print("2. Basic Statistics of Driving Records:")
pd.options.display.float_format = '{:,.2f}'.format
print(driving_csv.describe())
print("---------------------------------------")
print("3. Types of Input Columns from Driving Records:")
print(driving_csv.dtypes)
print("---------------------------------------")

## Driving records with break/zero speed should be eliminated
## I play the game with almost car stop!

# Extract center image(CI) and steering Angle(SA) from IMG folder & driving records
CI_path = [driving_csv.loc[i][0] for i in range(len(driving_csv))]
SA = [driving_csv.loc[i][3] for i in range(len(driving_csv))]
CI_path=np.array(CI_path)
SA = np.array(SA).astype(np.float32)

# Import images
CI_images = [mpimg.imread(imgpath) for imgpath in CI_path]


X_train = np.copy(driving_csv['Center Image']+'?'+driving_csv['Left Image']+'?'+driving_csv['Right Image'])
y_train = np.copy(driving_csv['Steering Angle']).astype(np.float32)
# Split data into Train & test data (90%:10%)
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

# Convert list type into ndarray for analysis convenience
#X_train = np.array(X_train)
#X_test = np.array(X_test)
#y_train = np.array(y_train)
#y_test = np.array(y_test)

print("7. Train_Test Split from original dataset:")
print("X_train Shape:",np.array(X_train).shape)
print("y_train Shape:",np.array(y_train).shape)
print("X_test Shape:",np.array(X_eval).shape)
print("y_test Shape:",np.array(y_eval).shape)
print("---------------------------------------")

#Model Initializations in Keras
model = Sequential()
model.add(MaxPooling2D(pool_size=pool_size,input_shape=input_shape))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1164))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1))
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
model.summary()

model.fit_generator(image_generator(X_train, y_train, rows, cols, channel),samples_per_epoch=len(X_train)/batch_size, 
                    nb_epoch=nb_epoch,
                    validation_data=image_generator(X_eval, y_eval, rows, cols, channel),
                   nb_val_samples=int(len(X_train)/(batch_size*10)))

model_reps = model.to_json()

# Save the trained model
with open(OutputModelJson,'w' ) as f:
    json.dump(model_reps, f)
    model.save_weights(OutputModelH5)

