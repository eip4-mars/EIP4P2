Group Members : Bikash Bhoi, Deepak Gowtham

#### Hosted lambda function Link Here: [http://www.tsaimars.com.s3-website.ap-south-1.amazonaws.com/](http://www.tsaimars.com.s3-website.ap-south-1.amazonaws.com/)
#### Front-End Files : [https://github.com/eip4-mars/eip4-mars.github.io](https://github.com/eip4-mars/eip4-mars.github.io)

## Human Pose Estimation

#### Steps:
- Click on the [Link](http://www.tsaimars.com.s3-website.ap-south-1.amazonaws.com/)
- Click on "Human Pose Estimation" on top Tab
- Warmup lambda if needed by clicking "Warm Up Lambda" and wait ~40 Seconds for Lambda Function to be ready
- Upload the File
- Click on "Estimate Pose" Button

#### Screenshot
---------
![](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/hpe_UI.jpg)

## Notebook : 
[Notebook](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/Human_Pose_Estimation.ipynb)

## Model Description : 

The model used here is a Headless Resnet50 with Deconvolution Block to extract body landmarks.

Draw.io can be imported from : [xml file](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/Posenet_drawio.xml)


### Smaller BottleNeck Block (Residual Blocks)

![](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/bottleneck.jpg)

### ResNet50 Headless architecture:
Above Residual Blocks are used in this network architecture.

![](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/headlessResnet50.jpg)

### Deconvolution Block:
Output of ResNet50 Headless block is fed to this below block:

![](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/deconvBlock.jpg)


## JointsMSELoss:
JointsMSELoss is the average Mean Squared Error Loss between Predicted Heatmap and Ground Truth Heatmap across all Joints.

In our case there are 16 Joints So,

![](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/JointMSELoss.jpg)
