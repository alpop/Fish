#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:27:24 2017

@author: alexander
"""

#import IPython.display as dp
from PIL import Image
import os
import json


size = 768,432


species = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
work_dir = '/media/SANDISK/Fish/'
os.chdir(work_dir)

for spec in species:
  print spec  
  input_dir =  work_dir + 'train/' + spec + '/'
  output_dir = work_dir + 'train_thumb/' + spec +'/'
  json_file =  spec +  '_labels.json'

  #Read json
  with open(input_dir+ json_file) as json_data:
      d = json.load(json_data)

  for i in range(len(d)):
    # Crop image
    fn =  str(d[i]['filename'])    
    im = Image.open(os.path.join(input_dir,fn))
    scale =   float(size[1]) / im.size[1]       # scale to width 
    aspect =  im.size[0] / float(im.size[1])    # keep aspect ratio
    #print im.size, aspect, scale
    im = im.resize((int(size[1]*aspect),size[1]), Image.ANTIALIAS)
    im = im.crop((0,0,size[0],size[1])) 
    thumb_path = output_dir + fn
    im.save(thumb_path)
    # Modify bounding boxes
    for j in range(len(d[i]['annotations'])):
      d[i]['annotations'][j]['height'] = d[i]['annotations'][j]['height'] * scale
      d[i]['annotations'][j]['width'] = d[i]['annotations'][j]['width'] * scale       
      d[i]['annotations'][j]['x'] = d[i]['annotations'][j]['x'] * scale
      d[i]['annotations'][j]['y'] = d[i]['annotations'][j]['y'] * scale
    
  #Save to json
  with open(output_dir + spec +'_thumb_labels.json', 'w') as fp:
     json.dump(d, fp,indent=0)

#Create labels for DetectNet


for spec in species:
  print spec  
  input_dir =  work_dir + 'train_thumb/' + spec + '/'
  output_dir = work_dir + 'train_thumb_all_labels/'
  json_file =  spec +  '_thumb_labels.json'

  #Read json
  with open(input_dir+ json_file) as json_data:
      d = json.load(json_data)
  for i in range(len(d)):
    #One Label file per one image
    fn =  str(d[i]['filename'])  
    fn, ext = os.path.splitext(fn)
    with open(output_dir + fn + '.txt', 'w') as fp:
        # Convert annotations to required format
        for j in range(len(d[i]['annotations'])):            
            l = d[i]['annotations'][j]['x']
            t = d[i]['annotations'][j]['y'] 
            r = l + d[i]['annotations'][j]['width']
            b = t + d[i]['annotations'][j]['height'] 
    
            if spec != 'NoF': 
               type = 'Car' 
               truncated = 0
               occluded  = 3
               alpha  = 0
               tail = '0 0 0 0 0 0 0 0'
            else:
                type ='DontCare'
                truncated = -1
                occluded  = -1
                alpha  = -10
                tail = '-1 -1 -1 -1000 -1000 -1000 -10'

            label = type +' '+             \
                    str(truncated) +' '+   \
                    str(occluded) +' '+    \
                    str(alpha) +' '+       \
                    str(l) +' '+ str(t) +' '+ str(r) +' '+ str(b) +' '+ tail
      
      
            fp.write(label + '\n')


'''
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
Example:
Car 0.00 0 1.57 571.58 178.70 595.09 199.68 1.41 1.59 4.47 -1.93 1.85 51.47 1.53
DontCare -1 -1 -10 622.06 177.10 634.60 187.56 -1 -1 -1 -1000 -1000 -1000 -10
                     
'''

