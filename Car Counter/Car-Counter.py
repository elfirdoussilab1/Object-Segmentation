from ultralytics import YOLO
import cv2
import cvzone
from sort import *

# Colors: RBG: R (0, 0, 255) / G (0, 255, 0) / B (255, 0, 0)
# Connect to the webcam
cap = cv2.VideoCapture('../Yolo/Videos/cars.mp4')

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
count = 0

# Importing the mask
mask = cv2.imread("mask.png")

# Tracking
# max_age : limit of frames that is gone and still recognize it as a region
tracker = Sort(max_age= 20, min_hits= 3, iou_threshold = 0.3)

limits = [350, 297, 673, 297]

while True:
    # Reading image on the webcam
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

            # Fancy rectangle
            w , h = x2 - x1, y2 - y1

            # confidence with 2 decimals
            conf = round(box.conf[0].item(), 2)
            # Class name
            cls = classNames[int(box.cls[0].item())]
            if cls == "car" and conf > 0.3:
                # displaying the class if it's a car
                #cvzone.putTextRect(img, text=cls + "  " + str(conf) , pos=(max(0, x1), max(40, y1)), scale=1,
                                   #thickness=2)
                #cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=8, rt = 5)
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
        cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

        cx, cy = x1 + w//2 , y1 + h//2
        cv2.circle(img, center = (cx, cy), radius = 5, color = (255, 0, 255))
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            count += 1


    # displaying image
    cv2.imshow("Image", img)
    cv2.waitKey(1) # 1 ms delay