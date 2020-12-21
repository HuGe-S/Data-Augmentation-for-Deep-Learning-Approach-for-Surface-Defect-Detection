
# Data Augmentation of Deep-Learning-Approach-for-Surface-Defect-Detection
  This repository is built upon the Tensorflow implementation of "**Segmentation-Based Deep-Learning Approach for Surface-Defect Detection**"(https://github.com/Wslsdx/Deep-Learning-Approach-for-Surface-Defect-Detection) to apply multiple data augmentation methods on this model. The purpose of this repository is to find the best data augmentation method that improve the SDD model's perfromance.
  
  We use "**imgaug**" (https://github.com/aleju/imgaug) to implement the  various data augmentation method. You can go [here](https://github.com/aleju/imgaug/blob/master/README.md) to find out how to use imgaug.
  
  The author submitted the paper to  Journal of Intelligent Manufacturing (https://link.springer.com/article/10.1007/s10845-019-01476-x), where it was published In May 2019 . 
  
# The test environment
```
python 3.6
cuda 9.0
cudnn 7.1.4
Tensorflow 1.12
Pillow 7.0
Spicy 1.10
```
# Dataset

  The dataset we use is from the paper, you can download [KolektorSDD](https://www.vicos.si/Downloads/KolektorSDD) here.
  It consisits of appoximately 300 electronic commutator's surface images. About 20 % of these images have defect.
  You can refer to the [paper](https://link.springer.com/article/10.1007/s10845-019-01476-x) for details of the experiment.
 


# Repository useage
  To simplify the usage of the model, we build a Google colab environment to help users quickly build the framework and implement their own data augmentation method.
  There are 4 main setp to test your framework after  clone this repository in Google Colab
  
  1.Environment installization
  
  2.Data Augmentation method implementation
  
  3.Model Training
  
  4.Visualization using tensorboard.
  
  Each step is clearly marked in the google colab environment. To start your experiment [JUST CLICK HERE](https://colab.research.google.com/drive/1z4N_2Zc2vaYxTC1UekVL3mHfoWTylUJo?usp=sharing) or upload (https://github.com/HuGe-S/Deep-Learning-Approach-for-Surface-Defect-Detection/blob/master/Deep_Learning_Approach_for_Surface_Defect_Detection.ipynb) To your colab.
  
# Things you should know after training

  There are few folder that will be autimaticlly created once you finished training. Checkpoint is where the model is stored, Visualization is where the images during training are stored, And in Log folder you can find how the loss value converges over time.
  
  We also track the loss value using tensorboard. Here is one example:
  
  ![example](https://github.com/HuGe-S/Deep-Learning-Approach-for-Surface-Defect-Detection/blob/master/images/i1.JPG)
  
# Example of logfile
```
[INFO]   2020-12-01 15:59:40,093    start testing
[INFO]   2020-12-01 16:00:30,770     total number of samples = 159
[INFO]   2020-12-01 16:00:30,771    positive = 21
[INFO]   2020-12-01 16:00:30,771    negative = 138
[INFO]   2020-12-01 16:00:30,771    TP = 19
[INFO]   2020-12-01 16:00:30,771    NP = 1
[INFO]   2020-12-01 16:00:30,771    TN = 137
[INFO]   2020-12-01 16:00:30,771    FN = 2
[INFO]   2020-12-01 16:00:30,771    accuracy(准确率) = 0.9811
[INFO]   2020-12-01 16:00:30,771    prescision（查准率） = 0.9500
[INFO]   2020-12-01 16:00:30,771    recall（查全率） = 0.9048
[INFO]   2020-12-01 16:00:30,771    the visualization saved in ./visualization/test
```

# Addtional cmd

## testing the KolektorSDD
  After downloading the KolektorSDD and changing the param[data_dir]
  ```
  python run.py --test
  ```
  Then you can find the result in the "/visulaiation/test" and  "Log/*.txt"
  
 ## training the KolektorSDD
 
 **First, only the segmentation network is independently trained, then the weights for the segmentation network are frozen and only the decision network layers are trained.**
 
   training the segment network
   ```
   python run.py --train_segment
   ```
   training the  decision network
   ```
   python run.py  --train_decision
   ```
   training the total network( not good）
   ```
   python run.py  --train_total
   ```
 
