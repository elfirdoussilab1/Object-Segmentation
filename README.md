# Object-Segmentation

In this repository, we apply object-segmentation algorithms to different tasks like counting objects of the same class that appear in an image or a video. The main methods that have been used to do so include the object detection algorithm YOLO (You Only Look Once), and a file called **sort** that attributes to each object an id.

To be able to run all these experiments, you should preferably work with Python version 3.10x, and install all the required libraies that you can find in the file:

```sh
requirements.txt
```

## Yolo:
This folder contains several examples of usages of the algorithm YOLO, for example, the file `Yolo-Images` shows you how to apply this object-segmentation algorithm for images, and `Yolo-Videos` applies it for some examples of videos, and finally `Yolo-Webcam` classifies all what's on your webcam.

![Example-yolo](images/example-yolo.png)

## Car Counter:
In the car counter file, we implement an algorithm that counts how many different cars traverse a road. For this purpose, we use mask to not consider some regions of the image, and to count cars that cross a red line (that becomes green when a car goes on it).

![Example-car-counter](images/example-car-counter.png)

## People counter:
We implemented also an algorithm that counts people going up or down on an esacalator. When their center cross the red line, they get counted.

![Example People counter](images/example-people-counter.png)

## Personal Equipment prediction:
For this project, I retrained the YOLO model on the construction safety equipments databse that we can find in Roboflow: [Construction Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28). We want to predict whether or not a person is wearing these equipments, and if so we drawn a green rectangle, otherwise we draw a red one.

![Example PPE](images/example-ppe.png)

Enjoy !

