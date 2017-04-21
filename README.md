# Behavioral Cloning

---

[//]: # (Image References)
[ARC]: ./Summary/architecture.PNG
[IDR]: ./Summary/InputDrivingRecords.PNG
[SAC]: ./Summary/SACount.PNG
[SALC]: ./Summary/SAlogct.PNG
[SA]: ./Summary/steeringangle.PNG
[ri]: ./Summary/recoveryimage.gif
[sd]: ./Summary/simulatedriving.gif

#### The goals / steps of this project are the following:
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

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop very deep layers combo and trial-error tuned parameters. I learnt a lesson from project 2 that too shallow of a cnn might not be able to achieve a nice test accuracy and, to batch trial-and-error tuning parameters are needed, no matter how experienced I am at mastering this art.

My first step was to use a convolution neural network model similar to the one I used in project 2, I thought this model might be appropriate because it achieved very nice test accuracy(around 97%) with a very good performance. But it does not achieve the same result in this case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added 2 dropout layers and tested on the proportions of trian-test split rate.

The final step was to run the simulator to see how well the car was driving around track one. Initially, I stayed my car in the very center of the road and I was very happy about how nice a game driver I am. But there were a few spots where the vehicle fell off the track when I use my model to drive it. To improve the driving behavior in these cases, I re-simulate the data with more scenarios that car goes back to the road from the edges of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 1.1 Training data

To capture good behaviour of driving, training data was chosen to keep the vehicle driving on the road, as well as several recovery from the lanes. I used a combination of center lane driving, recovering from the left and right sides of the road. A simple comprehensive analysis for the center lane driving was made to check if the data met a minimum requirements:

![alt text][IDR]

The four figures inspects closely how the four indicators(**Steering Angle**, **Throttle**, **Break** and **Speed**) change over time. I tried to avoid staying my game car in the center all the time, so that my model learns little from the **homogeneous** data. As indicated from the four, 
* Steering angle keeps changing between left and right
* Throttle has been used several times
* Break has never been used
* Speed is staying around 30 miles per hour and has **ONE natural stop**, which is okay.

The following figures randomly picks steer angle and visulize what it means for different steer angles:

![alt text][SA]

Next, I looked at two histograms of steering angle to check how often the car has been staying in the center and doing nothing.

![alt text][SAC]

![alt text][SALC]

Although for the majority of time my car is driving in the center, but the logcount figure shows that car is steering left and right with different degrees evenly, mostly caused by recovery from the left & right lanes, which looks like this:

![alt text][ri]

#### 1.2 Model arcthiecture

My model consists of 5 convolution neural network with various filter sizes and depths. The model also includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

The final model architecture is indicated as the following:

![alt text][ARC]

#### 1.3 Overfitting in the model

The model contains 5 dropout layers in order to reduce overfitting. All of them are deployed after the transoformations of cnn layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. But the data comes from the same track in the simulator - track one. 

The model has been tested by running it through the simulator more than 10 times and ensuring that the vehicle could stay on the track. Many trial-and-error process has been applied to tune the hyper-parameters and adjustments on the model architecture(overfitting-reduction layers, cnn layers and etc.)

Also, data set randomly shuffled the 10% of the data into a validation set to determine if the model was over or under fitting.

#### 1.4 Model parameter tuning

The model used an adam optimizer with learning rate 0.001 (determined by manual tunning after more than 50 experiments), so the learning rate was not tuned manually.

