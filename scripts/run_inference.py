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
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    result_dict = {}
    model = init_detector(args.config, args.checkpoint, device=args.device)
    #print(model)
    # test a single image
    count = 0
    for img in os.listdir(args.img):
        print(img)
        result = inference_detector(model, args.img + '/' + img)
        #print(result[1])
        result_dict[img] = result[1].tolist()
        #print(result_dict)
        count+=1
    outfile = open("result_dense_cans.json", "w")
    json.dump(result_dict, outfile)
        # show the results
        #show_result_pyplot(model, args.img+'/'+ img, result, score_thr=args.score_thr)
        #if count==5:
        #    break



async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
