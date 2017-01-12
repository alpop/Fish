#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Detectnet/Caffe model prediction example
Fish detection
Created on Sun Jan  11 00:41:52 2017

@author: alexander
"""
#import IPython.display as dp
import datetime
import caffe
import os
import glob
import json
import ntpath

work_dir = '/media/SANDISK/Fish/'
os.chdir(work_dir)

#caffe_root = '/home/alexander/caffe'  # путь в корень каффе
#sys.path.insert(0, caffe_root + 'python')

#Set CPU or GPU mode for caffe
#caffe.set_mode_cpu() # CPU mode or GPU mode next:
caffe.set_device(0)
caffe.set_mode_gpu()
#Load model
model_dir = '/media/SANDISK/Fish/model4/'
model_def = model_dir + 'deploy.prototxt' #model description
model_weights = model_dir + 'snapshot_iter_1890.caffemodel' # weights
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#Get test imageS
image_dir  = '/media/SANDISK/Fish/test_stg1_thumb/'
images = glob.glob(image_dir +'*.jpg')


#Image preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# Reshape inputs 
net.blobs['data'].reshape(1,        # batch size 
                          3,        # 3-channel images
                          432,768) # image size is 432x768

start_ = datetime.datetime.today()
count = 0

json_data = []

for image in images:
    count +=1
    #img = Image.open(image) 
    #dp.display(img)
    img = caffe.io.load_image(image)# test image loading... 
    imageP = transformer.preprocess('data', img) # preprocess image
    #Predicting and printing results
    net.blobs['data'].data[0] = imageP # feed preprocessed image into the net
    preds = net.forward() # predicting
    results = preds['bbox-list'][0] # prediction results   
    #Prepare annotations for sloth
    print image
    im, ext = os.path.splitext(image)
    im = ntpath.basename(im) + '.jpg'
    
    templ = {          
             "class": "image",
             "filename": "jpg"
             } 
    templ['filename'] = im

    if ((int(results[0][0]) == 0) & (int(results[0][1]) == 0) &
        (int(results[0][2]) == 0) & (int(results[0][3]) == 0)):
       json_data.append(templ)
       continue
        
    for i in range(results.shape[0]):
        height = results[i][3]-results[i][1]
        width  = results[i][2]-results[i][0]
        if (height == 0) & (width == 0):
            break
        x = results[i][0]
        y = results[i][1]
        
        a = {
               "class": "rect",
               "height": 0,
               "width":  0,
               "x": 0,
               "y": 0
              }
    
        a ["height"] = float(height)
        a ["width"]  = float(width)
        a ["x"]  = float(x)
        a ["y"]  = float(y)
        
        if i == 0:
            templ["annotations"]=[]
        
        templ["annotations"].append(a.copy())
            
    json_data.append(templ) 

end_ = datetime.datetime.today()

print '\nStart:  ', start_
print 'End:    ',   end_
elapsed = end_ - start_
print 'Elapsed:', elapsed
print 'Speed (images per second):', round(count/elapsed.total_seconds(),2)


print 'Dumping json annotations for sloth...'
with open(image_dir + 'DetectNet_labels_model4.json', 'w') as fp:
     json.dump(json_data, fp,indent=0)
print 'Done.'

     




      

      




 

