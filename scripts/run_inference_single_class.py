import asyncio
import numpy as np
import os
from argparse import ArgumentParser
import json
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument(
        '--box-out-file', type=str, default='', help='json outfile for predicted boxes')
    parser.add_argument(
        '--mask-pred', type=bool, default=False, help='flag to denote if model predicts masks')
    parser.add_argument(
        '--mask-out-file', type=str, default='', help='json outfile for predicted masks')
    parser.add_argument(
        '--network', type=str, default='', help='network architecture')
    parser.add_argument(
        '--box-result-dir', type=str, default='', help='directory to store box jsons')
    parser.add_argument(
        '--mask-result-dir', type=str, default='', help='directory to store mask jsons')

    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    result_box_dict = {}
    result_mask_dict = {}
    model = init_detector(args.config, args.checkpoint, device=args.device)

    count = 0
    for img in os.listdir(args.img):
        print(img)
        result = inference_detector(model, args.img + '/' + img)
        if args.mask_pred == True:
            # yolact
            print(np.shape(np.vstack((result[0][0], result[0][1], result[0][2], result[0][3])))) 
            result_box_dict[img] = np.vstack((result[0][0], result[0][1], result[0][2], result[0][3])).tolist() #result[0][0].tolist()
            #result_mask_dict[img] = result[1][0]
        else:
            # retinanet
            result_box_dict[img] = result[0].tolist()

        count+=1
        if count == 100:
            break
  
    outfile_box = open(args.box_result_dir + '/' + args.box_out_file + '_' + args.network + '.json', "w")
    json.dump(result_box_dict, outfile_box)

    if args.mask_pred == True:
        outfile_mask = open(args.mask_result_dir + '/' + args.mask_out_file + '_' + args.network + '.json', "w")
        json.dump(result_mask_dict, outfile_mask)

if __name__ == '__main__':
    args = parse_args()
    main(args)
