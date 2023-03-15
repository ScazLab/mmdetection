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

NMS_THRESHOLD = 0.5

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

def convert_bbox_to_polygon(bbox):
    xmin, ymin, xmax, ymax = bbox
    tl = (xmin, ymin)
    tr = (xmax, ymin)
    br = (xmax, ymax)
    bl = (xmin, ymax)
    return Polygon([tl, tr, br, bl])

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

def convert_coords_to_polygon(rbox):
    """Convert rotated bounding box to Polygon. Order of the coordinates is important
    
    Args:
        rbox: (list) List of points of rotated bounding box coordinates
        
    Returns:
        A Shapely Polygon object
    """
    return Polygon(rbox)

def distance(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)

def nms_filter(scores, classes, boxes, nms_threshold, bbox_type):
    """
    NMS filter takes list of scores, classes and boxes as an input and
    returns filter scores, classes and boxes.
    """

    def overlap(box1, box2):
        try:
            intersect_area = float(box1.intersection(box2).area) 
            if intersect_area == 0.0:
                return 0.0
            return intersect_area / (box1.area + box2.area - intersect_area)
        except Exception as e:
            print(box1.area, box2.area, intersect_area)
            print(e)
            print(box1, box2)
            return 0.0
            
    prediction_ids = list(range(len(scores)))
    #print('boxes', boxes)
    predictions = sorted(zip(prediction_ids, scores, classes, boxes), key=lambda x: x[1])
    
    if bbox_type == "axis":
        # convert bbox to polygon
        boxes_poly = [convert_bbox_to_polygon(box) for box in boxes]
    else:
        # rotated boxes
        boxes_poly = [convert_coords_to_polygon(box) for box in boxes]
  
    
    #boxes_poly = [Polygon(box) for box in boxes]
    #print(predictions)
    scores = []
    classes = []
    boxes = []
    while len(predictions) > 0:
        cur_id, cur_score, cur_class, cur_box = predictions.pop()
        scores.append(cur_score)
        classes.append(cur_class)
        boxes.append(cur_box)
        predictions = [x for x in predictions if overlap(boxes_poly[cur_id], boxes_poly[x[0]]) < nms_threshold]
    
    scores, classes, boxes = map(np.array, (scores, classes, boxes))
    
    #print(boxes, scores)
    return scores, classes, boxes
    

def run_inference(model, image, num_classes):
    all_bboxes = []
    all_classes = []
    all_scores = []
    all_rboxes = []
    all_im_masks = []

    detections = inference_detector(model, image)
    '''
    detections have the following structure:
    a. It is a tuple with 2 elements
    b. element A: list of boxes, element B: list of masks
    c. Each list has 4 elements, each element correponding to each of the 4 categories

    (box_list, mask_list)
    box_list: [cat_1, cat_2,...., cat_4]
    mask_list: [cat_1, cat_2,...., cat_4]

    box_list[0]: np.ndarray (N,5) --> N is the number of category_1 boxes
    mask_list[0]: np.ndarray (N,720,1280) --> N is the number of category_1 masks

    convert each mask in that numpy array to rotated bbox
    '''
    box_list, mask_list = detections[0], detections[1]

    for cat_id in range(num_classes):
        scores = list()
        boxes, masks = box_list[cat_id], mask_list[cat_id]
        if boxes.shape[0]==0:
            continue
        '''
        Boxes shape: (N,5)
        Masks shape: (N, 720, 1280)
        N is the number of predictions for category cat_id for given image
        '''
    
        scores = boxes[:,4]
        # drop score from boxes and make it [x1,y1,x2,y2]
        boxes = boxes[:,:4]
    
        classes = [cat_id + 1]*boxes.shape[0]
        classes = np.array(classes)
        
        for index, mask in enumerate(masks): 
            rbox = convert_masks_to_rbbox(mask,boxes[index])
            if len(rbox)>0:
                for rb in rbox:
                    rb = np.array(rb)
                    all_rboxes.append(rb.tolist())
            else:
                all_rboxes.append(boxes[index])

        all_bboxes.extend(boxes.astype(int).tolist())
        all_classes.extend(classes.tolist())
        all_scores.extend(scores.tolist())
    #print(all_bboxes, all_classes, all_scores, all_rboxes)

    # filtered_scores, filtered_classes, filtered_bboxes =  nms_filter(all_scores, all_classes, all_bboxes, NMS_THRESHOLD, bbox_type='axis')
    # filtered_scores, filtered_classes, filtered_rboxes = nms_filter(all_scores, all_classes, all_rboxes, NMS_THRESHOLD, bbox_type='rotated')
    # print(filtered_rboxes, filtered_classes, filtered_scores)

    return (all_bboxes, all_classes, all_scores, all_rboxes)

def main():
    # mask threshold dis used for binarizing the mask. This will determine the quality of the mask obtained 
    mask_threshold = 0.5  
    num_classes = 8

    # exp_name
    #exp_name = "baseline_exp/"

    # Specify the path to model config and checkpoint file
    config_parent_path = '../configs/hide_and_seek/'
    model_parent_path = '~/catkin_ws/src/hide_and_seek/hr_hide_seek/models/robot_detection'

    config_file = join(config_parent_path, 'yolact_r101_1x8_robot_detection.py')
    checkpoint_file = join(model_parent_path, "epoch_100.pth")

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda')

    # root = '/home/dg777/project/recycling/Recycling_Dataset/v2/test/images/'
    # test_data = '/home/dg777/project/recycling/Recycling_Dataset/v2/test/annotations/test.json'

    root = '~/catkin_ws/src/hide_and_seek/hr_hide_seek/data/robot_detection_images/all_images/'
    test_data = '~/catkin_ws/src/hide_and_seek/hr_hide_seek/data/robot_detection_annotations/coco_json_heading_all.json'

    with open(test_data,'r') as f:
        test_json = json.load(f)
    #print(test_json.keys())

    annotations = pd.DataFrame.from_dict(test_json["annotations"])
    images = pd.DataFrame.from_dict(test_json["images"])
    categories = pd.DataFrame.from_dict(test_json["categories"])
    
    result_json = dict()
    times = list()
    count = 0
    #print(len(images))

    '''
    Get predictions for every image - bbox, rbbox, class, scores
    '''
    for index,row in enumerate(images.itertuples()):
        start_time = time.time()
        print(count)
        count = count + 1
        # import pdb; pdb.set_trace()
        image_name = row.file_name
        image_id = row.id
        im_height = row.height
        im_width = row.width
        image_name_key = image_name.split('/')[-1].split('.')[0]
        print("Image: ", root+image_name)
   
        image = load_image_into_numpy_array(root + image_name)
        print("Input Size:", image.shape)

        all_bboxes, all_classes, all_scores, all_rboxes = run_inference(model, image, num_classes)

        '''
        Process Ground Truth
        '''
        ann = annotations[annotations['image_id']==image_id]
            
        gt_boxes = list()
        gt_classes = list()
        gt_rbox = list()

        for index,a_row in enumerate(ann.itertuples()):
            #result_json = dict()
            #print(a_row)
            (xmin,ymin,xmax,ymax) = (a_row.bbox[0],a_row.bbox[1],a_row.bbox[0]+a_row.bbox[2],a_row.bbox[1]+a_row.bbox[3])
            # Check if annotations exceed the bounds of the image 
            if xmax > im_width:
                xmax = im_width-1
            if ymax > im_height:
                ymax = im_height-1

            new_box = [xmin,ymin,xmax,ymax]
            gt_boxes.append(new_box) 
            gt_classes.append(a_row.category_id)
               
            seg_xs = a_row.segmentation[0][0::2]
            seg_ys = a_row.segmentation[0][1::2]

            gt_seg = list()
            for x,y in zip(seg_xs,seg_ys): 
                gt_seg.append([int(x),int(y)])
            gt_rbox.append(gt_seg)

            result_json[image_name_key] = {
                "detection_boxes": all_bboxes,
                "detection_classes": all_classes,
                "detection_scores": all_scores,
                "detection_rboxes":all_rboxes,
                "gt_boxes":gt_boxes,
                "gt_classes": gt_classes,
                "gt_count": len(gt_boxes),
                "gt_rboxes": gt_rbox,
            }
            # print(result_json)
            #result_json = dict()
            #print(result_json)
    json_file = 'heading_test.json'
    with open(json_file,'w') as f:
        json.dump(result_json, f, indent=2)
            #print(result_json['dense_mix_6'])

if __name__ == '__main__':
    main()
