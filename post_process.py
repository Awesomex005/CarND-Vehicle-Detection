import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label


def filter_bbox_by_size(bbox_list):
    # the vehicle should have a relatively larger appearance if it is close to us
    valid_bboxes = []
    threshold_size = 96*96
    minimum_size = 50*50
    for bbox in bbox_list:
        square = (bbox[1][0]-bbox[0][0])*(bbox[1][1]-bbox[0][1])
        if square < minimum_size:
            continue
        if bbox[1][1] > 550:
            # the vehicle is close to us
            if square > threshold_size:
                valid_bboxes.append(bbox)
        else:
            valid_bboxes.append(bbox)
    return valid_bboxes
    
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, (bbox[0][0],bbox[0][1]), (bbox[1][0],bbox[1][1]), color, thick)
    return imcopy

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
    
def apply_threshold(heatmap, threshold):
    heatmap[heatmap < threshold] = 0
    return heatmap
    
def find_labeled_bboxes(labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(box)
    return boxes

