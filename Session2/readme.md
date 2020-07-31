Group Members : Deepak Gowtham, Bikash Bhoi



#### 1. Explaining the code:

Code Link : [Notebook](https://github.com/eip4-mars/EIP4P2/blob/master/Session2/EVA4P2_Session2_Mobilenetv2_custom.ipynb)

##### Steps: 

- Total Number of images against classes: 20732
  - Flying Birds : 8318
  - Large QuadCopters : 4821
  - Small QuadCopters : 3613
  - Winged Drones : 3980
- Zipped the Images to Google drive and unizipped it directly to Colab Local
- Created Custom Dataloader to write  test.csv and train.csv with image path and label [With 80-20 Split]
- Store Label details in labels.csv
- Used Train and test transforms and loaders with parametrized batch Size




#### 2. Resizing Strategy

- Images were directly read in Train/Test Loader
- Resized the Image using transforms.Resize(224), Which will set the lesser dimension of the image to 224 Keeping the same aspect ratio
- On the Resized image, did a CentrreCropp of 224x224
- Applied other augmentation on top of the 224x224 image



#### 3. Model :

- Used MobilenetV2 with Imagenet weights
- Changed last FC Layer to 1280 -> 4 from 1280 -> 1000
- added Log_softmax()
- Used LR_Finder to Find the Best LR for OneCycleLR Scheduler



#### 4. Accuracy vs Epochs graphs for train and test curves

![Accuracy](https://github.com/eip4-mars/EIP4P2/blob/master/Session2/accuracy.jpg)



#### 5. 10 misclassified images for each of the classes as an Image Gallery

![Class_0](https://github.com/eip4-mars/EIP4P2/blob/master/Session2/Misclassified_0.jpg)

![Class_1](https://github.com/eip4-mars/EIP4P2/blob/master/Session2/Misclassified_1.jpg)

![Class_2](https://github.com/eip4-mars/EIP4P2/blob/master/Session2/Misclassified_2.jpg)

![Class_3](https://github.com/eip4-mars/EIP4P2/blob/master/Session2/Misclassified_3.jpg)

