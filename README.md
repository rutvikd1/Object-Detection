# Object-Detection

Object detection is a essential task for autonomous vehicles. This repository presents code and configurations for detecting objects from a vehicles perspective in urban environement utilizing the [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md). The object detection model is trained on 2D boding boxes from the nuScenes dataset for 4 categories: vehicle, pedestrian, truck and bicycle.

<p align="center">
  <img src="/experiments/ssd-7layers/inference/output.gif" width="40%" />
  <img src="/experiments/ssd-7layers/inference/output-2.gif" width="40%" />
</p>

<!-- <div align='center'>
    <img src = >
    <img src = >
</div> -->

# Setup

The provided Dockerfile allows a hassle free setup of the environemnt. 

Use the following command to build the docker image from dockerfile 
```bash
docker build -t project-tfod:<version> -f Dockerfile .
```

To create and run the container a shell script is provided. This automatically mounts the GPU and the work directory to the docker container.

```bash
./run.sh
```

# About the dataset

For this project, the images from the forward facing camera of the dataset are utilized which contain different driving scenarios. A few examples of images used are shown below:
![image](/images/gt_boxes.png)

# Class distribution

The following diagram presents the category distribution of of the dataset. We consider mainly three categories, that include car, truck, pedestrian and bicycle.
![image](/images/pie_chart.png)

The analysis is performed in the v1.0-mini dataset, which contains fewer images. Due to this the data below looks sparse.

Further we analyze the categorywise bouding boxes for aspect ratio and area contained by the bouding boxes.

![image](/images/area_hist.png)

It could be seen that the area covered by most of the bounding boxes is 0.04. 

# Aspect Ratio distribution 

![image](/images/aspect_ratio_dist.png)

The image above represents the frequency of aspect ratio of the bouding boxes. The aspect ratios are all below 2.0, and hence this value was used for anchor box values in the training config. 

# Positional distribution of object.
The positional distributuion of object in the dataser is analyzed using a heatmap for all four categories.
![image](/images/heatmap.png)


This project performs 2D object detection on nuimages dataset from nuScenes. Implementing a SSD architecture with a Resnet50 backbone

