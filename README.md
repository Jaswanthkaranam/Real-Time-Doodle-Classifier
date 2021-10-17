# **REAL TIME DOODLE CLASSIFIER**
This doodle classifier is based on a convolutional neural network which is trained on 20 different classes. It guesses the doodle simultaneously when the user is drawing.


# **Dataset**
The neural network was trained on a data set of 1964925 images and then tested on a data set of 842452 images.

# **Libraries Utilized**
* Pytorch - for implementation of neural networks(consists of all the functions that will be needed to calculate various parameters)
* Numpy 
* OpenCV - for implementing a virtual drawing pad
* Matplotlib - for plotting graphs and showing images
 
# **CNN Model**
## **Architecture**

***Convolution Layer***

| Layers   | Kernel size | Filters | Maxpool | stride|
| -------- | --------    | ------- | ------- | ------ |
| conv1    | (5,5) | 6 | (2,2)   | 1      |
| conv2    | (5,5)       | 16       | (2,2)   | 1      |

***Fully Connected Layer***

| Layer | Size |
| -------- | -------- | 
| FC1    | 16x4x4, 120 | 
| FC1    | 120, 84| 
| FC1    | 84, 20     | 

***Hyperparameters***
| Parameter | Value |
| ---- | --- |
| Learning rate | 0.03 |
| Epochs | 100 |
|  Batch Size | 1024 | 
| Beta| 0.9 |
| Optimizer| SGD |
# **Output**
A plot of losses vs the number of epoch.
![](https://i.imgur.com/qTqufMC.png)
| Dataset | Accuracy |
| ---- |---- |
| Train| 91.1178 |
| Test | 90.3160 |


## **Few images of output**
![](https://i.imgur.com/VmgAGuX.png)

![](https://i.imgur.com/6Gf1A6m.png)







