#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 18-5-17 下午6:08
     # @Author  : Awiny
     # @Site    :
     # @Project : activity_recognition
     # @File    : __init__.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
from . import models
from . import utils
from . import data
from . import motion_convnet
from . import spatial_convnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning