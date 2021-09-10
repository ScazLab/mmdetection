from mmdet.apis import init_detector, inference_detector
import mmcv
from os.path import join
import json
import torch

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = (20, 20)

from ast import literal_eval
from shapely.geometry import Polygon, Point
from datetime import datetime
from PIL import ImageColor
import numpy as np
from glob import glob
import cv2
from itertools import compress
import numpy as np
import pandas as pd
import os
import json
from PIL import ImageColor
from os.path import join
import time
from copy import deepcopy

def mask_to_polygons(mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        cnts = res[-2]
        object_list = list()
        
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y = Polygon(box.tolist()).exterior.coords.xy
            coords = list([int(cx), int(cy)] for cx, cy in zip(x, y))
            if len(coords) == 5:
                object_list.append(coords)
            else:
                print("Error: ",len(coords), coords)
                # backup axis aligned bbox
                #this is done to keep rboxes size same as classes/bboxes/scores which is required for evaluation
            break
        return object_list, has_holes

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #return np.array(Image.open(path))

def convert_bbox_polygon(bbox):
    """Format bbox coordinates into Polygon
    
    Args:
        bbox: (list) List of axis-aligned bounding box coordinates in [xmin, ymin, xmax, ymax] order
    
    Returns:
        A list of coordinates of four corners of the bounding boxes
        
    """
    xmin, ymin, xmax, ymax = bbox
    tl = (xmin, ymin)
    tr = (xmax, ymin)
    br = (xmax, ymax)
    bl = (xmin, ymax)
    return [tl,tr,br,bl,tl]

def convert_masks_to_rbbox(im_mask,bbox):
    rbbox_list = list()
    rbbox, has_holes = mask_to_polygons(im_mask)
    if len(rbbox) > 0:
        for rb in rbbox:
            rbbox_list.append(rb)
            if len(rb) != 5:
                raise ValueError(len(rb), rb)
    else: #resort to axis boxes if rbox not possible from mask
        box_poly = convert_bbox_polygon(bbox)
        x, y = Polygon(box_poly).exterior.coords.xy
        coords = list([int(cx), int(cy)] for cx, cy in zip(x, y))
        rbbox_list.append(coords)
    #print(rbbox_list)
    return rbbox_list

def distance(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)

def run_inference():
    pass

def main():
    # mask threshold dis used for binarizing the mask. This will determine the quality of the mask obtained 
    mask_threshold = 0.5  
    num_classes = 4

    # exp_name
    #exp_name = "baseline_exp/"

    # Specify the path to model config and checkpoint file
    config_parent_path = '../configs/recycling/'
    model_parent_path = '../checkpoints/'

    config_file = join(config_parent_path, 'yolact_r101_1x8_recycling.py')
    checkpoint_file = join(model_parent_path, "epoch_55.pth")

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='gpu')

    root = '/home/dg777/project/recycling/Data/'
    test_data = '/home/dg777/project/recycling/Data/annotations/recycling_v1.json'

    with open(test_data,'r') as f:
        test_json = json.load(f)
    print(test_json.keys())

    annotations = pd.DataFrame.from_dict(test_json["annotations"])
    images = pd.DataFrame.from_dict(test_json["images"])
    categories = pd.DataFrame.from_dict(test_json["categories"])


if __name__ == '__main__':
    main()