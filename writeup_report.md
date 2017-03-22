# **Behavioral Cloning** 

---

** Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Overviews of the Project

---

#### 1. Project Files:
* model.py containing the script to create and train the model
* model.ipynb is the notebook of running the training model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Drive the car in the simulator with trained model
```sh
python drive.py model.h5
```

#### 3. Training the model

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model arcthiecture

My model consists of 5 convolution neural network with various filter sizes and depths.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Overfitting in the model

The model contains 5 dropout layers in order to reduce overfitting. All of them are deployed after the transoformations of cnn layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. But the data comes from the same track in the simulator - track one. 

The model has been tested by running it through the simulator more than 10 times and ensuring that the vehicle could stay on the track. Many trial-and-error process has been applied to tune the hyper-parameters and adjustments on the model architecture(overfitting-reduction layers, cnn layers and etc.)

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate 0.001(determined by manual tunning), so the learning rate was not tuned manually 

#### 4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develope very deep layers combo and trial-error tuned parameters. I learnt a lesson from project 2 that too shallow of a cnn might not be able to achieve a nice test accuracy and, to batch trial-and-error tuning parameters are needed, no matter how experienced I am at mastering this art.

My first step was to use a convolution neural network model similar to the one I used in project 2, I thought this model might be appropriate because it achieved very nice test accuracy(around 97%) with a very good performance. But it does not achieve the same result in this case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added 5 dropout layers and tested on the proportions of trian-test split rate.

The final step was to run the simulator to see how well the car was driving around track one. Initially, I stayed my car in the very center of the road and I was very happy about how nice a game driver I am. But there were a few spots where the vehicle fell off the track when I use my model to drive it. To improve the driving behavior in these cases, I re-simulate the data with more scenarios that car goes back to the road from the edges of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
maxpooling2d_1 (MaxPooling2D)    (None, 1, 53, 320)    0           maxpooling2d_input_1[0][0]       
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 1, 53, 320)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 1, 14, 5)      192005      lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 1, 14, 5)      0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 1, 7, 5)       4505        elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 1, 7, 5)       0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 1, 4, 5)       6005        elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 1, 4, 5)       0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 2, 3)       2883        elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 1, 2, 3)       0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 3)       1731        elu_4[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3)             0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 3)             0           flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 3)             0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          4656        elu_5[0][0]                      
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 1164)          0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      elu_6[0][0]                      
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 100)           0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        elu_7[0][0]                      
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 50)            0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         elu_8[0][0]                      
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 10)            0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          elu_9[0][0]                      
====================================================================================================
Total params: 333,856
Trainable params: 333,856
Non-trainable params: 0
