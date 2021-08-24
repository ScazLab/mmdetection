import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

result_dict = json.load(open('result_dense_cans.json'))
data_dir = '../../dense_cans/'
save_dir = 'Results_Dense_Cans/'
count=0
for image_name in os.listdir(data_dir):
    bboxes = result_dict[image_name]
    image = cv2.imread(data_dir+'/'+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image_name)
    print(len(bboxes))
    for bbox in bboxes:
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        confidence = bbox[4]
        box_length = bbox[2]-bbox[0]
        box_width = bbox[3]-bbox[1]
        if box_length >=25 and box_width >= 25:
            print(box_length, box_width, confidence)

            image = cv2.rectangle(image, start_point, end_point, color=(0,255,0), thickness=3)
            cv2.putText(image, str(confidence), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    count+=1
    #if count == 5:
    #    break
    #print(type(image))
    plt.imsave(save_dir+image_name, image)
