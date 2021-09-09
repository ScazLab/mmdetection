#!/usr/bin/env python

from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from os.path import join
import pandas as pd
import json
from matplotlib import cm
import cv2
from copy import deepcopy
from skimage.io import imread
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--image-path', type=str, default='', help='path to images')
    parser.add_argument(
        '--coco-json-path', type=str, default='', help='path to the COCO JSON')
    args = parser.parse_args()
    return args

def viz_dataset(args):
    test_data = args.coco_json_path
    root = args.image_path

    with open(test_data,'r') as f:
        test_json = json.load(f)
    train_annotations = pd.DataFrame.from_dict(test_json["annotations"])
    train_images = pd.DataFrame.from_dict(test_json["images"])
    train_categories = pd.DataFrame.from_dict(test_json["categories"])

    ids = list(train_categories['id'])
    names = list(train_categories['name'])
    cat_mapping  = {}
    for i,n in zip(ids,names):
        cat_mapping[i]=n

    for index,image in enumerate(train_images.itertuples()):
        image_file = image.file_name   
        image_id = image.id 
        print("loading img:", image_file)
        print("Image is ID: ", image_id)
        image = imread(root + image_file)  
        img_ann = train_annotations[train_annotations['image_id']==image_id]
        
        for index,ann in enumerate(img_ann.itertuples()):
            bbox = ann.bbox
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0]+bbox[2], bbox[1]+bbox[3])
            class_id = ann.category_id
            
            image = cv2.rectangle(image, start_point, end_point, color=(0,0,255), thickness=3)
            
            masks = np.array(ann.segmentation)
            masks = masks.reshape((-1, 1, 2))

            image = cv2.polylines(image, np.int32([masks[:, 0]]),isClosed=True, color=(255,0,0), thickness=5)
            
            cv2.putText(image,cat_mapping[class_id],start_point,cv2.FONT_HERSHEY_SIMPLEX,1,color=(255,255,0), thickness=2)
         
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.show()    

if __name__ == '__main__':
    args = parse_args()
    viz_dataset(args)





