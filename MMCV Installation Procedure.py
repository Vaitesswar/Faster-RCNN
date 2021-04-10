#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('conda install -c anaconda cudnn')

# install mmcv-full thus we could use CUDA operators
get_ipython().system('pip install mmcv-full==latest+torch1.6.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html # For pytorch (1.6.0) & CUDA (10.2)')

# Install mmdetection
get_ipython().system('git clone https://github.com/open-mmlab/mmdetection.git')
get_ipython().run_line_magic('cd', 'mmdetection')

get_ipython().system('pip install -r requirements/build.txt')
get_ipython().system('pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI')
get_ipython().system('pip install -v -e . # or "python setup.py develop"')

# To resolve the version conflicts of PILLOW, we reinstall PILLOW==7.0.0
get_ipython().system('pip install PILLOW==7.0.0')


# In[ ]:


# Split json file into train and test data
python './cocosplit.py' --having-annotations -s 0.9 './roboflow/train/_annotations.coco.json' './roboflow/train/train.json' './roboflow/test/test.json'

