from ultralytics import YOLO 
import cv2 
import cvzone 
import math 
 
  
model = YOLO("best.pt") 
  
classNames = ['ColonCancer'] 
  
prev_frame_time = 0 
new_frame_time = 0 
  
while True: 
    img = cv2.imread('Dataset\Test\images\Train_polyps_ (68).jpg') 
    results = model(img, stream=True) 
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
 
  
    cv2.imshow("Image", img) 
    cv2.waitKey(1)
