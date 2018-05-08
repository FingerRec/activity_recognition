#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# other util
# 计算第一个匹配和前五个匹配结果
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk) #最大k
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#利用torch保留模型，如果是最好则拷贝到model_best
def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

#写入csv文件
def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))      
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)   

#read flow
def read_flow(flow_or_path, quantize=False, *args, **kwargs):
    """Read an optical flow map
    Args:
        flow_or_path(ndarray or str): either a flow map or path of a flow
        quantize(bool): whether to read quantized pair, if set to True,
                        remaining args will be passed to :func:`dequantize_flow`
    Returns:
        ndarray: optical flow
    """
    if isinstance(flow_or_path, np.ndarray):
        if (flow_or_path.ndim != 3) or (flow_or_path.shape[-1] != 2):
            raise ValueError(
                'Invalid flow with shape {}'.format(flow_or_path.shape))
        return flow_or_path
    elif not isinstance(flow_or_path, str):
        raise TypeError(
            '"flow_or_path" must be a filename or numpy array, not {}'.format(
                type(flow_or_path)))

    if not quantize:
        with open(flow_or_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except:
                raise IOError('Invalid flow file: {}'.format(flow_or_path))
            else:
                if header != 'PIEH':
                    raise IOError(
                        'Invalid flow file: {}, header does not contain PIEH'.
                            format(flow_or_path))

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))
    else:
        from cvbase.image import read_img
        from cvbase.opencv import IMREAD_UNCHANGED
        dx_filename, dy_filename = _pair_name(flow_or_path)
        dx = read_img(dx_filename, flag=IMREAD_UNCHANGED)
        dy = read_img(dy_filename, flag=IMREAD_UNCHANGED)
        flow = dequantize_flow(dx, dy, *args, **kwargs)

    return flow.astype(np.float32)
# flow2img
def flow2img(flow, BGR=True):
    x, y = flow[:,:, 0], flow[:,:, 1]

    #plt.imshow(x,  cmap='Greys_r')
    #print x.shape
    #plt.imshow(y,  cmap='Greys_r')
    #plt.show()
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
    ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
    hsv[..., 0] = (an / 2).astype(np.uint8)
    hsv[..., 1] = (cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
    hsv[..., 2] = 255
    img = []
    if BGR:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
    else:
        img[0] = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
        img[1] = cv2.cvtColor(y, cv2.COLOR_HSV2RGB)
        return img