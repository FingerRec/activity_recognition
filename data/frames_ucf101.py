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
import cPickle as pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import os
import sys
import subprocess

def class_process(dir_path,  pickele_path):
  if not os.path.isdir(dir_path):
    return "not a dir"
  dict = {}
  for file_name in os.listdir(dir_path):
    if file_name == ".DS_Store": continue
    if '.bin' in file_name: continue
    video_dir_path = os.path.join(dir_path, file_name)
    image_indices = []
    for image_file_name in os.listdir(video_dir_path):
      if 'frame' not in image_file_name:
        continue
      image_indices.append(int(image_file_name[6:11]))

    if len(image_indices) == 0:
      print('no image files', video_dir_path)
      n_frames = 0
    else:
      image_indices.sort(reverse=True)
      n_frames = image_indices[0]
      print(video_dir_path, n_frames)
    dict[file_name+'.avi'] = n_frames

  p = pickle.dumps(dict)
  f1 = open(pickle_path, "at")
  pickle.dump(p, f1, True)
  f1.close()
    #    pickle.dump(video_dir_path, open(pickle_path, "w"), True)
#    pickle.dump(n_frames, open(pickle_path, "w"), True)


if __name__=="__main__":
  dir_path = "/数据库/UCF-101切片/tvl1_flow/u"
  pickle_path = "dic/frame_count2.pickle"
  class_process(dir_path, pickle_path)