# Object-Detection

Object detection is a essential task for autonomous vehicles. This repository presents code and configurations for detecting objects from a vehicles perspective in urban environement utilizing the [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md). The object detection model is trained on 2D boding boxes from the nuScenes dataset for 4 categories: vehicle, pedestrian, truck and bicycle.

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


This project performs 2D object detection on nuimages dataset from nuScenes. Implementing a SSD architecture with a Resnet50 backbone
