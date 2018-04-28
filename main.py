#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2018/4/18 10:18
     # @Author  : Awiny
     # @Site    :
     # @File    : load_ucf101_image_list.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""

from utils.utils import *
from data import spatial_dataloader
from config import opt


def help():
    '''
    打印帮助的信息： python file.py help
     '''

    print('''
   usage : python {0} <function> [--args=value,]
   <function> := train | test | help
   example: 
           python {0} train --env='env0701' --lr=0.01
           python {0} test --dataset='path/to/dataset/root/'
           python {0} help
   avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

def train():
    pass
def test():
    pass
def validate():
    pass
def average_fusion():
    rgb_preds = opt.rgb_preds # rgb预测序列
    opf_preds = opt.opf_preds  # opf预测序列

    with open(rgb_preds, 'rb') as f:
        rgb = pickle.load(f)
    f.close()
    # rgb:3783长度，字典格式，[101]维度每个维度保存输出值，最大为71，最小为-30；
    with open(opf_preds, 'rb') as f:
        opf = pickle.load(f)
    f.close()
    # opf:字典格式，3783长度，每个101维度
    dataloader = spatial_dataloader.spatial_dataloader(BATCH_SIZE = opt.batch_size, num_workers = opt.num_workers,
                                         path = opt.spatial_train_data_root,
                                         ucf_list = opt.ucf_list,
                                         ucf_split = opt.ucf_split)
    # 得到数据集
    train_loader, val_loader, test_video = dataloader.run()
    # 得到训练集，验证集，测试集
    # train_loader: pyTorch的dataloader类型，key为9537，每个keys 为{'视频名称  ？可数' } 训练数据：9537帧
    # val_loader: keys:71877, values:{list}: 每个对应的类别（71877）
    # test_video:3783长度字典，{.avi文件名称（对应图片文件夹）,标签类别}               测试数据：3783帧
    video_level_preds = np.zeros((len(rgb.keys()), opt.num_classfication))
    video_level_labels = np.zeros(len(rgb.keys()))  # 标签，长度为3783，指视频个数
    correct = 0
    ii = 0
    for name in sorted(rgb.keys()):
        r = rgb[name]  # 101维向量
        o = opf[name]  # 101维向量

        label = int(test_video[name]) - 1  # 得到标签

        video_level_preds[ii, :] = (r + o)  # 两个相加
        video_level_labels[ii] = label
        ii += 1
        if np.argmax(r + o) == label:  # argmax找到向量/矩阵中最大值的索引
            correct += 1

    video_level_labels = torch.from_numpy(video_level_labels).long()  # numpy转为tensor，3783
    print(video_level_labels)
    video_level_preds = torch.from_numpy(video_level_preds).float()  # 3783 x 101, 每个存储一个值 000-100

    top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))
    # top1:88.3162
    # top5:98.0439
    print("top1 score is: %f, top5 score is: %f"%(top1, top5))

# 平均融合模块
if __name__ == '__main__':
    average_fusion()
