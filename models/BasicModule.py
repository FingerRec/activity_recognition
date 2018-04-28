#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''''''''''''''''''''''''''''''''
     # @Time    : 2018/4/27 22:01
     # @Author  : Awiny
     # @Site    : 
     # @File    : BasicModule
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
'''''''''''''''''''''''''''''''''
import scipy.io
import os
import torch as t
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

class BasicModule(t.nn.Module):
   '''
   封装了nn.Module，主要提供save和load两个方法
   '''

   def __init__(self,opt=None):
       super(BasicModule,self).__init__()
       self.model_name = str(type(self)) # 模型的默认名字

   def load(self, path):
       '''
       可加载指定路径的模型
       '''
       self.load_state_dict(t.load(path))

   def save(self, name=None):
       '''
       保存模型，默认使用“模型名字+时间”作为文件名，
       如AlexNet_0710_23:57:29.pth
       '''
       if name is None:
           prefix = 'checkpoints/' + self.model_name + '_'
           name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
       t.save(self.state_dict(), name)
       return name