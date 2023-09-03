from ultralytics import YOLO
import cv2
import cvzone
from sort import *

# In this project, we will count the number of people traversing the escalator in the video: people.mp4

# Colors: RBG: R (0, 0, 255) / G (0, 255, 0) / B (255, 0, 0)
# Working with the video people.mp4
cap = cv2.VideoCapture('../Videos/people.mp4')

# Loading the model
model = YOLO("../YOLO-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
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

# Importing the mask
mask = cv2.imread("mask.png")

# Tracking
# max_age : limit of frames that is gone and still recognize it as a region
tracker = Sort(max_age= 20, min_hits= 3, iou_threshold = 0.3)

# The line that detects people to count
limits = [150, 370, 700, 370]

# list of people going up with the escalator
list_idx_up = []
# list of people going down with the escalator
list_idx_down = []

while True:
    # Reading image on the video
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)
    # Classifying what's on the image region
    results = model(img_region, stream = True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # getting the coordinates of boxes: list of length 4
            coordinates = box.xyxy.numpy()[0]
            # convert the coordinates to integers
            coordinates = list(map(int, coordinates))
            # getting each coordinate
            x1, y1, x2, y2 = coordinates

            # width and height of the rectangle
            w , h = x2 - x1, y2 - y1

            # confidence with 2 decimals
            conf = round(box.conf[0].item(), 2)
            # Class name
            cls = classNames[int(box.cls[0].item())]
            if cls == "person" and conf > 0.3:
                # Adding the observation to the list of detections
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))


    results_tracker = tracker.update(detections)
    cv2.line(img, limits[:2], limits[2:], color = (0, 0, 255))
    # Tracking
    for res in results_tracker:
        x1, y1, x2, y2, id = list(map(int, res))
        cvzone.putTextRect(img, text=cls + "  " + str(id), pos=(max(0, x1), max(40, y1)), scale=1,
                           thickness=2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=5, rt=1, colorR=(255, 0, 0))

        # finding the center of each rectangle (person)
        cx, cy = x1 + w//2 , y1 + h//2
        cv2.circle(img, center = (cx, cy), radius = 5, color = (255, 0, 255))

        # Counting people going up
        if limits[0] < cx < limits[2] and limits[1] < cy < limits[1] + 10:
            # If id is not in list up nor in down
            if list_idx_down.count(id) == 0 and list_idx_up.count(id) == 0:
                list_idx_up.append(id)

                # Turing the line to green when a car passes through it
                cv2.line(img, limits[:2], limits[2:], color=(0, 255, 0))

        # Counting people going down
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1]:
            # If id is not in list up nor in down
            if list_idx_down.count(id) == 0 and list_idx_up.count(id) == 0:
                list_idx_down.append(id)

                # Turing the line to green when a car passes through it
                cv2.line(img, limits[:2], limits[2:], color=(0, 255, 0))


    cvzone.putTextRect(img, text = f"Up : {len(list_idx_up)}", pos = (50, 50))
    cvzone.putTextRect(img, text=f"Down : {len(list_idx_down)}", pos=(50, 110))


    # displaying image
    cv2.imshow("People", img)
    cv2.waitKey(1) # 1 ms delay