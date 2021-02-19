#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:30:10 2021

@author: pc-neuron
From Adam Ligocky
"""

import glob 
import cv2
import csv
#import matplotlib.pyplot as plt
import tqdm
import os

gt = False
verze = 'index-dataset/'
experiment = 'mb2_varroa_2/' # bees_health_ill bees_varroas _varroa

IMAGES_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/Bee_project/new_no_index/images/det/'
#IMAGES_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/Bee_project/dataset-yolo/images/test/'
LABELS_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/algorithms/pytorch-ssd/eval_results/'+ verze + experiment + 'labels/'
OUTPUT_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/algorithms/pytorch-ssd/eval_results/'+ verze + experiment + 'images/'

## for ground truth
# gt = True
# experiment = 'labels_varroa/' # bees_health_ill bees_varroas _varroa

# IMAGES_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/Bee_project/dataset-yolo/images/test/'
# LABELS_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/Bee_project/dataset-yolo/' + experiment + 'test/'
# OUTPUT_FOLDER = '/media/pc-neuron/Data/Lukas/PHD_Industry_Anomaly/Bee_project/dataset-yolo/vis/' + experiment

# IMAGES = 'images/'
# LABELS = 'labels/'
    
input_ext = 'jpg'
output_ext = 'jpeg'
txt_ext = 'txt'

### varroa
if '_varroa' in experiment or 'edited' in experiment:
    def cls_to_color(cls):
        if cls == 1: # varrea
            return (0, 0, 255)
        else:
            return (255, 255, 255)

### all classes
if 'all' in experiment: 
    def cls_to_color(cls):
        if cls == 0: # bee_pl
            return (0, 255, 0)
        elif cls == 1: # bee_np
            return (0, 255, 255)
        elif cls == 2: # drone
            return (255, 0, 0)
        elif cls == 3: # queen
            return (255, 255, 0)
        elif cls == 4: # bee_ill
            return (255, 0, 255)
        elif cls == 5: # varea
            return (0, 0, 255)
        else:
            return (255, 255, 255)

### bees_health_ill
if 'bees_health_ill' in experiment or '640' in experiment: 
    def cls_to_color(cls):
        if cls == 1: # bee_helthy
            return (0, 255, 0)
        elif cls == 2: # bee_ill
            return (255, 0, 255)
        else:
            return (255, 255, 255)

### bees_varroas   
if 'bees_varroas' in experiment: 
    def cls_to_color(cls):
        if cls == 1: # bee
            return (0, 255, 0)
        elif cls == 2: # varea
            return (0, 0, 255)
        else:
            return (255, 255, 255)

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

cnt = 0
for img_path in tqdm.tqdm(sorted(glob.glob(IMAGES_FOLDER + '*.' + input_ext))):
    print(img_path)

    image = cv2.imread(img_path)
    (img_height, img_width, img_chann) = image.shape

    csv_path = img_path.replace(input_ext, txt_ext).replace(IMAGES_FOLDER, LABELS_FOLDER)
    if not os.path.exists(csv_path):
        final_img_path = img_path.replace(IMAGES_FOLDER, OUTPUT_FOLDER).replace(input_ext, output_ext)
        cv2.imwrite(final_img_path, image)
        continue

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
#             print('   ', row)

            cls = int(row[0])
            x1 = int(float(row[1])) 
            y1 = int(float(row[2]))
            x2 = int(float(row[3]))
            y2 = int(float(row[4]))
            
            if gt:
                cx = float(row[1])
                cy = float(row[2])
                w = float(row[3])
                h = float(row[4])
                
                x1 = int((cx - w/2) * img_width)
                x2 = int((cx + w/2) * img_width)
                y1 = int((cy - h/2) * img_height)
                y2 = int((cy + h/2) * img_height)

            color = cls_to_color(cls)
            start_point = (x1, y1)
            end_point = (x2, y2)
            thickness = 2
#             print(start_point, end_point, color, thickness)
            if color != None:
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

#         plt.imshow(image) 
#         plt.show()

        final_img_path = img_path.replace(IMAGES_FOLDER, OUTPUT_FOLDER).replace(input_ext, output_ext)
        cv2.imwrite(final_img_path, image)
