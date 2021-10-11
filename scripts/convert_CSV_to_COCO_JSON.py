#!/usr/bin/env python
import itertools
import pandas as pd
import json
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import box
from collections import Counter, OrderedDict, defaultdict
from shapely.validation import *
from iteration_utilities import unique_everseen
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--csv-path', type=str, default='', help='path to CSV files with annotations')
    parser.add_argument(
        '--json-save-path', type=str, default='', help='path to save the COCO JSON')
    args = parser.parse_args()
    return args

def get_filename_idx_map(filename_list):
    d_dict = defaultdict(lambda: len(d_dict))
    list_ids= [d_dict[n] for n in filename_list]
    file_id_map = defaultdict()
    for filename, idx in zip(filename_list, list_ids):
        file_id_map[filename] = idx+1
    return file_id_map

def CSV2JSON(args):
    csv_path = args.csv_path
    cat_map = {'Can': 1, 'Bottle':2, 'Milk Jug':3, 'Cardboard':4}
    data = pd.read_csv(csv_path)
    filename_list = []

    image_dict = {}
    image_dict_list = []

    ann_dict = {}
    ann_dict_list = []

    instance_id = 0

    for index, row in data.iterrows():
        # read all points and convert to [(x1, y1), (x2, y2), ...]
        points = row.values[5]
        points_dict = json.loads(points)
        x_list = points_dict['all_points_x']
        y_list = points_dict['all_points_y']
        xy_list = []
        for x,y in zip(x_list, y_list):
            xy_tuple = (x,y)
            xy_list.append(xy_tuple)

        # form a Shapely polygon
        polygon = Polygon(xy_list)
        polygon = polygon.buffer(0) # to make polygon valid
        
        # map filename to unique ID
        filename = row.values[0]
        print(filename)
        filename_list.append(filename)
        file_id_map = get_filename_idx_map(filename_list)
        image_id = file_id_map[filename]
        
        # get category from metadata and map to unique category ID
        meta_data = row.values[6]
        meta_data_dict = json.loads(meta_data)
        cat = meta_data_dict['Category']
        cat_id = cat_map[cat]

        if polygon.is_valid:
            # obtain axis-aligned bbox from polygon
            x0, y0, x1, y1 = polygon.bounds
            bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            bbox_area = (x1 - x0) * (y1 - y0)
            #print(filename, image_id, instance_id, bbox, polygon, cat, cat_id)
            
            # IMPORTANT: Check for Multipolygons. If found, correct annotations. 
            if polygon.geom_type == 'MultiPolygon':
                print(filename, instance_id, cat, row)
            
            # Create annotation dict for COCO JSON
            ann_dict["id"] = instance_id
            ann_dict["image_id"] = image_id
            ann_dict["category_id"] = cat_id
            ann_dict["segmentation"] = [list(itertools.chain(*list(polygon.exterior.coords)))]
            ann_dict["bbox"] = bbox
            ann_dict["area"] = bbox_area
            ann_dict["iscrowd"] = 0
            ann_dict_list.append(ann_dict)
            ann_dict = {}
            instance_id += 1
        
        # Create image dict for COCO JSON
        image_dict["id"] = image_id
        image_dict["file_name"] = filename
        image_dict["width"] = 1280
        image_dict["height"] = 720
        image_dict["date_captured"] = ""

        image_dict_list.append(image_dict)
        image_dict = {}

    # convert image_dict_list to contain only unique entries
    image_dict_list = list(unique_everseen(image_dict_list))

    # Create category dict for COCO JSON
    cat_dict = {}
    cat_dict_list = []
    for name, idx in cat_map.items():
        cat_dict["supercategory"] = "Recyclables"
        cat_dict["id"] = int(idx)
        cat_dict["name"] = name
        cat_dict_list.append(cat_dict)
        cat_dict = {}

    # Assemble final COCO JSON
    COCO_JSON_dict = {}
    COCO_JSON_dict["images"] = image_dict_list
    COCO_JSON_dict["annotations"] = ann_dict_list
    COCO_JSON_dict["categories"] = cat_dict_list

    JSON_filename = args.json_save_path
    with open(JSON_filename, "w") as f:
        json.dump(COCO_JSON_dict, f)
    print("COCO JSON Saved!!")

if __name__ == '__main__':
    args = parse_args()
    CSV2JSON(args)


