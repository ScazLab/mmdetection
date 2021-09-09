import numpy as np
import cv2
import json
import os
import re
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--box-result-json-path', type=str, default='', help='json path for predicted boxes from run_inference')
    parser.add_argument(
        '--data-dir', type=str, default='', help='path where the images are located')
    parser.add_argument(
        '--crops-save-dir', type=str, default='', help='path to directory where crops are saved')
    args = parser.parse_args()
    return args

def bbox2crop(args):
    '''
    Expected JSON format: {'name_frameID:[[[x_min, y_min], [x_max, y_max]], [[x_min, y_min], [x_max, y_max]], ......]'}
    '''
    crops = json.load(open(args.box_result_json_path))
    data_dir = args.data_dir
    crop_results_dir = args.crops_save_dir

    if not os.path.exists(crop_results_dir):
        os.mkdir(crop_results_dir)

    count = 0
    for img_name in os.listdir(data_dir):
        image = cv2.imread(data_dir + '/' + img_name)
        frame_id = re.findall(r'[A-Za-z]+|\d+', img_name)[1]
        for k, coords in crops.items():
            if k == img_name:
                for i in range(len(coords)):
                    x_min = coords[i][0][0]
                    y_min = coords[i][0][1]
                    x_max = coords[i][1][0]
                    y_max = coords[i][1][1]
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    crop_name = 'frame_' + str(frame_id) + '_crop_'+str(i)+'.png' # add timestamp
                    cv2.imwrite(os.path.join(crop_results_dir, crop_name), cropped_image)
                count += 1

if __name__ == '__main__':
    args = parse_args()
    bbox2crop(args)
