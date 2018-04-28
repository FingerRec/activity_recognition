#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''''''''''''''''''''''''''''''''
     # @Time    : 2018/4/16 21:46
     # @Author  : Awiny
     # @Site    : 
     # @File    : n_frames_ucf101_hmdb51.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
'''''''''''''''''''''''''''''''''
from __future__ import print_function, division
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import os
import sys
import subprocess

def class_process(dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  for file_name in os.listdir(class_path):
    video_dir_path = os.path.join(class_path, file_name)
    image_indices = []
    for image_file_name in os.listdir(video_dir_path):
      if 'image' not in image_file_name:
        continue
      image_indices.append(int(image_file_name[6:11]))

    if len(image_indices) == 0:
      print('no image files', video_dir_path)
      n_frames = 0
    else:
      image_indices.sort(reverse=True)
      n_frames = image_indices[0]
      print(video_dir_path, n_frames)
    with open(os.path.join(file_name, 'n_frames'), 'w') as dst_file:
      dst_file.write(str(n_frames))


if __name__=="__main__":
  dir_path = sys.argv[1]
  for class_name in os.listdir(dir_path):
    class_process(dir_path, class_name)