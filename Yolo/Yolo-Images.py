from ultralytics import YOLO
import cv2

# Downloading YOLO (model) weights
model = YOLO('../YOLO-weights/yolov8l.pt')

# Object segmentation results
results = model("../images/1.png", show = True)

# Stopping the process to see the image classified
cv2.waitKey(0)
