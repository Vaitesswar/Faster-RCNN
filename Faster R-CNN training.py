#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from RoboDataset import RoboFlowDataset


# In[ ]:


# Model training
import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import copy
import os.path as osp

# Train detector for 50, 100 & 150 epochs
epochs = [50,100,150]
checkpoint = 'Faster R-CNN pretrained weights.pth'
        
for i in range(len(epochs)):
    # Choose to use a config and initialize the detector
    config = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'

    cfg = Config.fromfile(config)

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 11

    # If we need to finetune a model based on a pre-trained detector, we need to use load_from to set the path of checkpoints. 
    cfg.load_from = checkpoint
    print(checkpoint)

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    factor = 8 + (8*i) # Reduce learning rate by a factor of 2 every 50 epochs
    cfg.optimizer.lr = 0.02 / factor
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10
    cfg.total_epochs = 50
    cfg.lr_config.step = list() # Learning rate schedule

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 250
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 250

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Modify dataset type and path
    cfg.dataset_type = 'RoboFlowDataset'
    cfg.data_root = 'roboflow'

    cfg.data.train.type = 'RoboFlowDataset'
    cfg.data.train.data_root = 'roboflow/train/'
    cfg.data.train.ann_file = 'train.json'
    cfg.data.train.img_prefix = ''

    cfg.data.val.type = 'RoboFlowDataset'
    cfg.data.val.data_root = 'roboflow/test/'
    cfg.data.val.ann_file = 'test.json'
    cfg.data.val.img_prefix = ''

    cfg.data.test.type = 'RoboFlowDataset'
    cfg.data.test.data_root = 'roboflow/test/'
    cfg.data.test.ann_file = 'test.json'
    cfg.data.test.img_prefix = ''

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    classes = ('biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green','trafficLight-GreenLeft','trafficLight-Red',
              'trafficLight-RedLeft','trafficLight-Yellow','trafficLight-YellowLeft','truck')

    # Add an attribute for visualization convenience
    model.CLASSES = classes

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    train_detector(model, datasets, cfg, distributed=False, validate=True)

    # Save the model weights
    state = {
        'epoch': epochs[i],
        'state_dict': model.state_dict()
        }
    checkpoint = 'Faster R-CNN weights ' + str(epochs[i]) + ' epochs' + '.pth.tar'
    torch.save(state, checkpoint)

