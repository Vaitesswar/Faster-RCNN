#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from RoboDataset import RoboFlowDataset
import mmcv
from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py' 

# Setup a checkpoint file to load
epoch = 50
checkpoint = 'Faster R-CNN weights ' + str(epoch) + ' epochs' + '.pth.tar' # Load the checkpoint weight file

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)

# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# modify num classes of the model in box head
config.model.roi_head.bbox_head.num_classes = 11

# Initialize the detector
model = build_detector(config.model, test_cfg=config.test_cfg)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
classes = ('biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green','trafficLight-GreenLeft','trafficLight-Red',
          'trafficLight-RedLeft','trafficLight-Yellow','trafficLight-YellowLeft','truck')

model.CLASSES = classes

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)

# Convert the model into evaluation mode
model.eval()


# In[ ]:


# Visualization of result for a single test image
from mmdet.apis import inference_detector, show_result_pyplot
img = 'roboflow/test/1478897374542789638_jpg.rf.49cfdb590a022bf82f47dda02560e597.jpg'
img = mmcv.imread(img)
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.9)


# In[ ]:


# Inference on all test images
import os
import json
import numpy as np
import time

data = json.load(open('roboflow/test/test.json'))
image_dict = data['images']
annot_dict = data['annotations']
annotations = list()
det_results = list()

start = time.time()

for i in range(1):
    image_path = os.path.join('roboflow/test/',image_dict[i]['file_name'])
    image = mmcv.imread(image_path)
    if np.all(image) != None: # Ensure only image file is read
        
        # Storing predicted results 
        # list(list): outer list indicates images, and the inner list indicates per-class detected bboxes        
        result = inference_detector(model, image)
        det_results.append(result)
    
        # Storing groud truth
        # list[dict]: Each element in list is an image and the keys of dict are bboxes (n,4 np array) and labels (n, np array)
        bbox = list()
        label = list()
        indices = list()
        ID = image_dict[i]['id']
        for j in range(len(annot_dict)):
            if annot_dict[j]['image_id'] == ID:
                # Need to convert from COCO format (xmin,ymin,width,height) to Pascal VOC dataset format (xmin,ymin,xmax,ymax)
                bboxes = annot_dict[j]['bbox']
                bboxes[2] = bboxes[2] + bboxes[0]
                bboxes[3] = bboxes[3] + bboxes[1]
                bbox.append(bboxes)
                label.append(annot_dict[j]['category_id']-1) # Need to -1 since MAP function assumes label starts from 0
                indices.append(j)

        for index in sorted(indices, reverse=True):
            del annot_dict[index]
        
        bbox = np.array(bbox)
        label = np.array(label)
        gt_dict = dict()
        gt_dict['bboxes'] = bbox
        gt_dict['labels'] = label
        annotations.append(gt_dict)
        
end = time.time()
print(end - start)


# In[ ]:


# Compute average MAP for different IOU threshold
from mmdet.core.evaluation.mean_ap import *
from mmdet.core.evaluation.bbox_overlaps import *

eval_map(det_results,
         annotations,
         scale_ranges=None,
         iou_thr=0.5,
         dataset=None,
         logger=None,
         nproc=1)

