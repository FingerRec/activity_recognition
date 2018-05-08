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
# ------------------------------------------------ Global ------ -------------------------------------------------------
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


# ------------------------------------------------ Data Set PATH -------------------------------------------------------
    project_path = '/home/zdwyf1/Code/ActivityRecognition/activity_recognition/' #工程路径
    spatial_train_data_root = '/home/zdwyf1/DataSet/jpegs_256/'  # 空间域网络训练集存放路径
    temporal_train_data_root = '/home/zdwyf1/DataSet/tvl1_flow/'  # 时域训练集存放路径

    spatial_test_data_root = '/home/zdwyf1/DataSet/jpegs_256/'  # 测试集存放路径
    temporal_test_data_root = '/home/zdwyf1/DataSet/tvl1_flow/'  # 测试集存放路径
    ucf_list = project_path + 'data/UCF_list/'# 标签数据位置
    ucf_split = '01' # ucf_split 序号
    frame_count = project_path + 'data/dic/frame_count.pickle'  # 帧统计序列
    dir_flow_path = "/media/zdwyf1/Elements/OpticalFlow/data/flow/"
    dir_grayimg_path = "/media/zdwyf1/Elements/OpticalFlow/data/grayimg/"
# ------------------------------------------------ BasicModule PATH ----------------------------------------------------
    BasicModule_spatial_checkpoint_path = project_path + 'checkpoints/BasicModule/spatial_checkpoint.pth.tar'
    BasicModule_temporal_checkpoint_path = project_path + 'checkpoints/BasicModule/temporal_checkpoint.pth.tar'

    BasicModule_spatial_best_model_path = project_path + 'checkpoints/BasicModule/spatial_model_best.pth.tar'
    BasicModule_temporal_best_model_path = project_path + 'checkpoints/BasicModule/temporal_model_best.pth.tar'

    # 预测序列位置
    BasicModule_rgb_preds = project_path + 'data/record/BasicModule/spatial/spatial_video_preds.pickle'  # rgb预测序列
    BasicModule_opf_preds = project_path + 'data/record/BasicModule/motion/motion_video_preds.pickle'  # opf预测序列

    #
    BasicModule_rgb_train_record_path = project_path + 'data/record/BasicModule/spatial/rgb_train.csv'
    BasicModule_rgb_test_record_path = project_path + 'data/record/BasicModule/spatial/rgb_test.csv'
    BasicModule_flow_train_record_path = project_path + 'data/record/BasicModule/motion/opf_train.csv'
    BasicModule_flow_test_record_path = project_path + 'data/record/BasicModule/motion/opf_test.csv'

# ------------------------------------------------ ResNet101 PATH ------------------------------------------------------
    ResNet101_spatial_checkpoint_path = project_path + 'checkpoints/ResNet101/spatial_checkpoint.pth.tar'
    ResNet101_temporal_checkpoint_path = project_path + 'checkpoints/ResNet101/temporal_checkpoint.pth.tar'

    ResNet101_spatial_best_model_path = project_path + 'checkpoints/ResNet101/spatial_model_best.pth.tar'
    ResNet101_temporal_best_model_path = project_path + 'checkpoints/ResNet101/temporal_model_best.pth.tar'

    # 预测序列位置
    ResNet101_rgb_preds = project_path +'data/record/ResNet101/spatial/spatial_video_preds.pickle'  # rgb预测序列
    ResNet101_opf_preds = project_path +'data/record/ResNet101/motion/motion_video_preds.pickle'  # opf预测序列

    #
    ResNet101_rgb_train_record_path = project_path + 'data/record/ResNet101/spatial/rgb_train.csv'
    ResNet101_rgb_test_record_path = project_path + 'data/record/ResNet101/spatial/rgb_test.csv'
    ResNet101_flow_train_record_path = project_path + 'data/record/ResNet101/motion/opf_train.csv'
    ResNet101_flow_test_record_path = project_path + 'data/record/ResNet101/motion/opf_test.csv'

# ------------------------------------------------ bninception PATH ----------------------------------------------------
    bninception_spatial_checkpoint_path = project_path + 'checkpoints/bninception/spatial_checkpoint.pth.tar'
    bninception_temporal_checkpoint_path = project_path + 'checkpoints/bninception/temporal_checkpoint.pth.tar'

    bninception_spatial_best_model_path = project_path + 'checkpoints/bninception/spatial_model_best.pth.tar'
    bninception_temporal_best_model_path = project_path + 'checkpoints/bninception/temporal_model_best.pth.tar'

    # 预测序列位置
    bninception_rgb_preds = project_path +'data/record/bninception/spatial/spatial_video_preds.pickle'  # rgb预测序列
    bninception_opf_preds = project_path +'data/record/bninception/motion/motion_video_preds.pickle'  # opf预测序列

    #
    bninception_rgb_train_record_path = project_path + 'data/record/bninception/spatial/rgb_train.csv'
    bninception_rgb_test_record_path = project_path + 'data/record/bninception/spatial/rgb_test.csv'
    bninception_flow_train_record_path = project_path + 'data/record/bninception/motion/opf_train.csv'
    bninception_flow_test_record_path = project_path + 'data/record/bninception/motion/opf_test.csv'

# ------------------------------------------------ Inceptionv4 PATH ----------------------------------------------------
    Inceptionv4_spatial_checkpoint_path = project_path + 'checkpoints/Inceptionv4/spatial_checkpoint.pth.tar'
    Inceptionv4_temporal_checkpoint_path = project_path + 'checkpoints/Inceptionv4/temporal_checkpoint.pth.tar'

    Inceptionv4_spatial_best_model_path = project_path + 'checkpoints/Inceptionv4/spatial_model_best.pth.tar'
    Inceptionv4_temporal_best_model_path = project_path + 'checkpoints/Inceptionv4/temporal_model_best.pth.tar'

    # 预测序列位置
    Inceptionv4_rgb_preds = project_path +'data/record/Inceptionv4/spatial/spatial_video_preds.pickle'  # rgb预测序列
    Inceptionv4_opf_preds = project_path +'data/record/Inceptionv4/motion/motion_video_preds.pickle'  # opf预测序列

    #
    Inceptionv4_rgb_train_record_path = project_path + 'data/record/Inceptionv4/spatial/rgb_train.csv'
    Inceptionv4_rgb_test_record_path = project_path + 'data/record/Inceptionv4/spatial/rgb_test.csv'
    Inceptionv4_flow_train_record_path = project_path + 'data/record/Inceptionv4/motion/opf_train.csv'
    Inceptionv4_flow_test_record_path = project_path + 'data/record/Inceptionv4/motion/opf_test.csv'

# ------------------------------------------------ InceptionV3 PATH ----------------------------------------------------
    # pretrained on kinetics
    Inceptionv3_spatial_checkpoint_path = project_path + 'checkpoints/Inceptionv3/inception_v3_kinetics_rgb_pretrained.zip'
    Inceptionv3_temporal_checkpoint_path = project_path + 'checkpoints/Inceptionv3/inception_v3_kinetics_flow_pretrained.zip'

    Inceptionv3_spatial_best_model_path = project_path + 'checkpoints/Inceptionv3/spatial_model_best.pth.tar'
    Inceptionv3_temporal_best_model_path = project_path + 'checkpoints/Inceptionv3/temporal_model_best.pth.tar'

    # 预测序列位置
    Inceptionv3_rgb_preds = project_path + 'data/record/Inceptionv3/spatial/spatial_video_preds.pickle'  # rgb预测序列
    Inceptionv3_opf_preds = project_path + 'data/record/Inceptionv3/motion/motion_video_preds.pickle'  # opf预测序列

    #
    Inceptionv3_rgb_train_record_path = project_path + 'data/record/Inceptionv3/spatial/rgb_train.csv'
    Inceptionv3_rgb_test_record_path = project_path + 'data/record/Inceptionv3/spatial/rgb_test.csv'
    Inceptionv3_flow_train_record_path = project_path + 'data/record/Inceptionv3/motion/opf_train.csv'
    Inceptionv3_flow_test_record_path = project_path + 'data/record/Inceptionv3/motion/opf_test.csv'


# ------------------------------------------------ resnet101_tsn PATH ----------------------------------------------------
    resnet101_tsn_spatial_checkpoint_path = project_path + 'checkpoints/resnet101_tsn/inception_v3_kinetics_rgb_pretrained.zip'
    resnet101_tsn_temporal_checkpoint_path = project_path + 'checkpoints/resnet101_tsn/inception_v3_kinetics_flow_pretrained.zip'

    resnet101_tsn_spatial_best_model_path = project_path + 'checkpoints/resnet101_tsn/spatial_model_best.pth.tar'
    resnet101_tsn_temporal_best_model_path = project_path + 'checkpoints/resnet101_tsn/temporal_model_best.pth.tar'

    # 预测序列位置
    resnet101_tsn_rgb_preds = project_path + 'data/record/resnet101_tsn/spatial/spatial_video_preds.pickle'  # rgb预测序列
    resnet101_tsn_opf_preds = project_path + 'data/record/resnet101_tsn/motion/motion_video_preds.pickle'  # opf预测序列

    #
    resnet101_tsn_rgb_train_record_path = project_path + 'data/record/resnet101_tsn/spatial/rgb_train.csv'
    resnet101_tsn_rgb_test_record_path = project_path + 'data/record/resnet101_tsn/spatial/rgb_test.csv'
    resnet101_tsn_flow_train_record_path = project_path + 'data/record/resnet101_tsn/motion/opf_train.csv'
    resnet101_tsn_flow_test_record_path = project_path + 'data/record/resnet101_tsn/motion/opf_test.csv'

    resnet101_tsn_flow_origin_path = '/media/zdwyf1/Elements/OpticalFlow/data/flow/'

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