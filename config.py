#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''''''''''''''''''''''''''''''''
     # @Time    : 2018/4/27 21:42
     # @Author  : Awiny
     # @Site    : 
     # @File    : config.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
'''''''''''''''''''''''''''''''''
import scipy.io
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'ResNet101'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    spatial_train_data_root = '/数据库/UCF-101切片/jpegs_256/'  # 空间域网络训练集存放路径
    temporal_train_data_root = '/数据库/UCF-101切片/jpegs_256/'  # 时域训练集存放路径

    spatial_test_data_root = '/数据库/UCF-101切片/tvl1_flow/'  # 测试集存放路径
    temporal_test_data_root = '/数据库/UCF-101切片/tvl1_flow/'  # 测试集存放路径

    spatial_checkpoint_path = 'checkpoint/spatial_checkpoint.pth.tar'
    temporal_checkpoint_path = 'heckpoint/temporal_checkpoint.pth.tar'


    spatial_best_model_path = 'checkpoints/spatial_model_best.pth.tar'  # 加载预训练的模型的路径，为None代表不加载
    temporal_best_model_path = 'checkpoints/temporal_model_best.pth.tar'

    ucf_list = 'data/UCF_list/'#标签数据位置
    ucf_split = '01' #ucf_split 序号

    #预测序列位置
    rgb_preds = 'data/record/spatial/spatial_video_preds.pickle'  # rgb预测序列
    opf_preds = 'data/record/motion/motion_video_preds.pickle'  # opf预测序列
    frame_count = 'data/dic/frame_count.pickle' #帧统计序列

    #
    rgb_train_record_path = 'data/record/spatial/rgb_train.csv'
    rgb_test_record_path = 'data/record/spatial/rgb_test.csv'
    flow_train_record_path = 'data/record/motion/opf_train.csv'
    flow_test_record_path = 'data/record/motion/opf_test.csv'

    num_classfication = 101 #行为类别数目
    batch_size = 1  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    momentum = 0.9
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()