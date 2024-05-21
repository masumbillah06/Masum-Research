import subprocess
import sys

# Function to install a package using pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import sklearn and install if not available
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    print("scikit-learn not found. Installing...")
    install('scikit-learn')
    from sklearn.metrics import precision_score, recall_score, f1_score

from ultralytics import YOLO
import cv2
import cvzone
import math
import os
import numpy as np

# Load the YOLO model
model = YOLO("best.pt")

classNames = ['ColonCancer']

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    inter_area = inter_width * inter_height
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Function to load ground truth labels
def load_ground_truth(folder_path, image_file):
    label_file = os.path.join(folder_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(label_file):
        return []
    with open(label_file) as f:
        lines = f.readlines()
    ground_truths = []
    for line in lines:
        cls_id, x_center, y_center, width, height = map(float, line.strip().split())
        x1 = (x_center - width / 2) * img.shape[1]
        y1 = (y_center - height / 2) * img.shape[0]
        x2 = (x_center + width / 2) * img.shape[1]
        y2 = (y_center + height / 2) * img.shape[0]
        ground_truths.append((int(cls_id), x1, y1, x2, y2))
    return ground_truths

# Function to evaluate the model
def evaluate_model(model, folder_path, classNames, iou_threshold=0.5):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    y_true = []
    y_pred = []

    for image_file in image_files:
        img = cv2.imread(os.path.join(folder_path, image_file))
        ground_truths = load_ground_truth(folder_path, image_file)
        
        results = model(img)
        
        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                predictions.append((cls, conf, x1, y1, x2, y2))
        
        matched_gt = set()
        for gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 in ground_truths:
            y_true.append(gt_cls)
            best_iou = 0
            best_pred = None
            for i, (pred_cls, conf, pred_x1, pred_y1, pred_x2, pred_y2) in enumerate(predictions):
                iou = calculate_iou((gt_x1, gt_y1, gt_x2, gt_y2), (pred_x1, pred_y1, pred_x2, pred_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_pred = (pred_cls, conf)
            if best_iou >= iou_threshold:
                y_pred.append(best_pred[0])
                matched_gt.add(best_pred)
            else:
                y_pred.append(None)
        
        for pred in predictions:
            if pred not in matched_gt:
                y_pred.append(pred[0])
                y_true.append(None)

    # Filter out None values for true positives and predictions
    y_true_filtered = [label for label in y_true if label is not None]
    y_pred_filtered = [pred for pred in y_pred if pred is not None]

    if len(y_true_filtered) == 0:
        print("No ground truth labels found. Cannot calculate metrics.")
        return

    accuracy = sum(1 for x, y in zip(y_true_filtered, y_pred_filtered) if x == y) / len(y_true_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

# Evaluate the model
evaluate_model(model, 'Dataset/Test/images', classNames)
