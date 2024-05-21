from ultralytics import YOLO 
import cv2 
import cvzone 
import math 
import os

model = YOLO("best.pt") 

classNames = ['ColonCancer'] 

folder_path = 'Dataset\Test\images'

image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    img = cv2.imread(os.path.join(folder_path, image_file))
    
    # Perform object detection
    results = model(img, stream=True) 
    
    # Process detection results
    for r in results: 
        boxes = r.boxes 
        for box in boxes: 
            # Bounding Box 
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            
            w, h = x2 - x1, y2 - y1 
            cvzone.cornerRect(img, (x1, y1, w, h)) 
            # Confidence 
            conf = math.ceil((box.conf[0] * 100)) / 100 
            # Class Name 
            cls = int(box.cls[0]) 
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1) 

    # Display the image with detections
    cv2.imshow("Image", img) 
    cv2.waitKey(0) 
