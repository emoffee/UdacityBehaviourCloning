# **Behavioral Cloning** 

---

** Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/EDA1.png "Model Visualization"
[image2]: ./examples/EDA2.png "Grayscaling"
[image3]: ./examples/EDA3.png "Recovery Image"
[image4]: ./examples/EDA4.png "Recovery Image"


## Overviews of the Project

---

#### 1. Project Files:
* model.py containing the script to create and train the model
* model.ipynb is the notebook of running the training model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Drive the car in the simulator with trained model
```
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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develope very deep layers combo and trial-error tuned parameters. I learnt a lesson from project 2 that too shallow of a cnn might not be able to achieve a nice test accuracy and, to batch trial-and-error tuning parameters are needed, no matter how experienced I am at mastering this art.

My first step was to use a convolution neural network model similar to the one I used in project 2, I thought this model might be appropriate because it achieved very nice test accuracy(around 97%) with a very good performance. But it does not achieve the same result in this case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added 5 dropout layers and tested on the proportions of trian-test split rate.

The final step was to run the simulator to see how well the car was driving around track one. Initially, I stayed my car in the very center of the road and I was very happy about how nice a game driver I am. But there were a few spots where the vehicle fell off the track when I use my model to drive it. To improve the driving behavior in these cases, I re-simulate the data with more scenarios that car goes back to the road from the edges of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes illustrated in the following table:


![alt text][image4]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded the first lap on track one using center lane driving. Here is 20 random sample images of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on the center of the road. 

![alt text][image3]
![alt text][image1]

After the collection process, I had 3528 data points. I finally randomly shuffled the data set and put 10% of the data(353) into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 100. I used an adam optimizer so that manually training the learning rate wasn't necessary.
