# Faster-RCNN

## Model
Faster RCNN is an object detection architecture presented by Ross Girshick, Shaoqing Ren, Kaiming He and Jian Sun in 2015, and is one of the famous object detection architectures that uses convolution neural networks. 

In this project, the standard feature extractor in faster RCNN was replaced with feature pyramid network (FPN). The following link will be useful for understanding faster R-CNN with feature pyramid networks.

https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c

## Dataset
The dataset that was used in this project is the Udacity Self Driving Car Dataset which is an open-source dataset provided by Udacity, modified by Roboflow. The dataset includes driving in Mountain View California and neighboring cities during daylight conditions. It contains 97,942 labels across 11 classes and 15,000 images collected from a Point Grey research camera running at full resolution of 1920x1200 at 2 hz. The dataset was annotated by CrowdAI using a combination of machine learning and humans. There are 1,720 null examples (images with no labels). This dataset comprises of 11 classes namely car, pedestrian, traffic lightred, traffic light-green, truck, biker, traffic light, traffic lightred left, traffic light-green left, traffic light-yellow and traffic light- yellow left.

For the experiment conducted in this project, the dataset was split randomly in the ratio of 9:1 for training and test set. Validation set was not developed as the training process itself was computationally intensive. Manual validation was performed at regular interval of epochs.

## Instructions
1) The pre-trained model parameters of faster R-CNN with FPN trained on COCO 2017 dataset was used for fine tuning which was obtained from Open MMLab detection. Use MMCV Installation Procedure.py file to install the necessary packages. 
2) Split the dataset using cocosplit.py file
3) Train the model using Faster R-CNN training.py file and validate using Faster R-CNN validation.py file.

## Training approach
The optimizer used is stochastic gradient descent (SGD) with momentum of 0.9 and weight decay of 0.0001. The pretrained model was trained on 8 GPUs with an original learning rate of 0.02. Since only 1 GPU was used in our training, the learning rate was reduced by 8-fold to 0.0025. The learning rate was halved every 50 epochs and the input images were passed in batches of 2. The total number of epochs needed for optimal performance is 150. The loss is aggregate of two losses: cross entropy loss and L1 loss.

## Results
Mean Average Precision (mAP) values are computed for IOU spanning 0.5 to 0.95. Faster-RCNN gives a mAP of 38.14 % and detection speed of 21 frames per second. A sample result of a frame is shown below.

![download](https://user-images.githubusercontent.com/81757215/114263303-45265f00-9a17-11eb-90ca-1cf05ef1a078.png)
