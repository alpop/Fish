#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:27:24 2017

@author: alexander

Converts DetectNet labels to json Sloth

"""

#import IPython.display as dp

import os
import json
import glob
import pandas as pd
import ntpath
import sys

work_dir = '/media/SANDISK/Fish/train_thumb/'
os.chdir(work_dir)
input_dir =  work_dir + 'Y_train/'
output_dir = work_dir 


file_list = (glob.glob(input_dir+'*.txt')) # DetectNet labels

json_data = []

for fn in file_list:
      img, ext = os.path.splitext(fn)
      #img = ntpath.basename(img) + '.png'
      img = ntpath.basename(img) + '.jpg'
      templ =    {
        "annotations": [
            {
                "class": "rect",
                "height": 0,
                "width":  0,
                "x": 0,
                "y": 0
            }
        ],
        "class": "image",
        "filename": "jpg or png"
      }
    
      templ['filename'] = img
      print img
      try:
        #Read label:
        ann = pd.read_csv(fn,  parse_dates=True, header = None, names = None,sep=' ')
        #Annnotations loop:
        for i in range(len(ann)):
            templ['annotations'][i]['height'] = ann[7][i] - ann[5][i]
            templ['annotations'][i]['width']  = ann[6][i] - ann[4][i]
            templ['annotations'][i]['x']  = ann[4][i]
            templ['annotations'][i]['y']  = ann[5][i]
            if i < len(ann) - 1:
                templ['annotations'].append(templ['annotations'][i].copy())
      except pd.io.common.EmptyDataError as err:
            err1 = sys.exc_info()[0]
            print err,err1
            pass
        
      json_data.append(templ)

with open(output_dir + 'DetectNet_labels.json', 'w') as fp:
     json.dump(json_data, fp,indent=0)
