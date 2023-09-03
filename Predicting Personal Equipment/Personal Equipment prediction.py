from ultralytics import YOLO
import cv2
import cvzone

# Connect to the webcam
cap = cv2.VideoCapture('../Videos/ppe-2.mp4')

# Loading the model
model = YOLO("./training/best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

custom_color = (255, 0, 0)
while True:
    # Reading video
    success, img = cap.read()
    # Classifying what's on the video
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

            # confidence with 2 decimals
            conf = round(box.conf[0].item(), 2)
            # Class name
            cls = classNames[int(box.cls[0].item())]
            # plotting the boxes in the image, such that if a safety element is not detected, the color of the box is red
            if 'NO' in cls:
                # red
                custom_color = (0, 0, 255)
            elif cls == 'Person':
                # Blue
                custom_color = (255, 0, 0)
            else:
                # Green
                custom_color = (0, 255, 0)
            cv2.rectangle(img, pt1= (x1, y1), pt2 = (x2, y2), color = custom_color, thickness= 2)

            # displaying the class and the confidence in the image
            cvzone.putTextRect(img, text = cls + "  " + str(conf), pos = (max(0, x1) , max(40, y1)), scale = 1, thickness=2)


    # displaying image
    cv2.imshow("Video", img)
    cv2.waitKey(1) # 1 ms delay