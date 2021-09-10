import json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
import os
import math

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--box-result-json-path', type=str, default='', help='json path for predicted boxes from run_inference')
    parser.add_argument(
        '--data-dir', type=str, default='', help='path where the images are located')
    parser.add_argument(
        '--save-dir', type=str, default='', help='path to directory where results are saved')
    parser.add_argument(
        '--save-box-coord-json-path', type=str, default='', help='json path for saving coordinates of box crops')
    parser.add_argument(
        '--box-result-dir', type=str, default='', help='directory to store box jsons')
    parser.add_argument(
        '--mask-result-dir', type=str, default='', help='directory to store mask jsons')
    parser.add_argument(
        '--min-box-length', type=int, default=5, help='minimum length of predicted box to consider')
    parser.add_argument(
        '--min-box-width', type=int, default=5, help='minimum width of predicted box to consider')
    parser.add_argument(
        '--max-box-length', type=int, default=1000, help='maximum length of predicted box to consider')
    parser.add_argument(
        '--max-box-width', type=int, default=720, help='maximum width of predicted box to consider')
    parser.add_argument(
        '--y-min', type=int, default=20, help='y coordinate of the top of the conveyor belt')
    parser.add_argument(
        '--y-max', type=int, default=700, help='y coordinate of the bottom of the conveyor belt')
    args = parser.parse_args()
    return args

def main(args):
    result_dict = json.load(open(args.box_result_json_path))
    data_dir = args.data_dir
    save_dir = args.save_dir
    count = 0
    box_output_json = args.save_box_coord_json_path
    crop_dict = {}
    for image_name in os.listdir(data_dir):
        if image_name not in result_dict.keys():
            continue
        bboxes = result_dict[image_name]
        image = cv2.imread(data_dir+'/'+image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image_name)
        #print(len(bboxes))
        crop_list = []
        for bbox in bboxes:
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            confidence = bbox[4]
            box_length = bbox[2]-bbox[0]
            box_width = bbox[3]-bbox[1]
            #print(bbox)
             
            if box_length >= int(args.min_box_length) and box_width >= int(args.min_box_width): # check for very small boxes
                if box_length <= int(args.max_box_length) and box_width <= int(args.max_box_width): # check for very large boxes
                    if bbox[1] > int(args.y_min) and bbox[3] < int(args.y_max): # check for boxes predicted outside the conveyor belt
                        print(box_length, box_width, confidence)
                        crop_list.append([start_point, end_point])
                        image = cv2.rectangle(image, start_point, end_point, color=(0,255,0), thickness=3)
                        cv2.putText(image, str(confidence), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        crop_dict[image_name] = crop_list

        count+=1
        #if count == 5:
        #    break
        #print(type(image))
        plt.imsave(save_dir+image_name, image)

    outfile = open(box_output_json, "w")
    json.dump(crop_dict, outfile)

if __name__ == '__main__':
    args = parse_args()
    main(args)
