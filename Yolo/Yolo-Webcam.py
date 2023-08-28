from ultralytics import YOLO
import cv2
import cvzone
import math

# Connect to the webcam
cap = cv2.VideoCapture(0)

# Set the width
cap.set(3, 1280)

# Set height
cap.set(4, 720)

# Loading the model
model = YOLO("../YOLO-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    # Reading image on the webcam
    success, img = cap.read()
    # Classifying what's on the webcam
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # getting the coordinates of boxes: list of length 4
            coordinates = box.xyxy.numpy()[0]
            # convert the coordinates to integers
            coordinates = list(map(int, coordinates))
            # gettign each coordinate
            x1, y1, x2, y2 = coordinates
            # plotting the boxes in the image
            cv2.rectangle(img, pt1= (x1, y1), pt2 = (x2, y2), color = (255, 0, 255), thickness= 3)

            # Facy rectangle
            w , h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, bbox = (x1, y1, w , h))

            # confidence with 2 decimals
            conf = round(box.conf[0].item(), 2)
            # Class name
            cls = classNames[int(box.cls[0].item())]
            # displaying the class and the confidence in the image
            cvzone.putTextRect(img, text = cls + "  " + str(conf), pos = (max(0, x1) , max(40, y1)))


    # displaying image
    cv2.imshow("Image", img)
    cv2.waitKey(1) # 1 ms delay