#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 18-5-8 下午1:37
     # @Author  : Awiny
     # @Site    :
     # @Project : activity_recognition
     # @File    : flow2img.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

from PIL import Image
from pylab import *
import numpy as np
import cvbase as cvb
from config import opt
import matplotlib.pyplot as plt
import cv2

def showrgbflow(flow_path):
    cvb.show_flow(flow_path)
    return
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

def flow2grayimg(real_flow):
    imgH = Image.fromarray(real_flow[:, :, 0])
    imgH = imgH.convert('L')
    imgV = Image.fromarray(real_flow[:, :, 1])
    imgV = imgV.convert('L')
    return [imgH, imgV]


def flow2rgbimg(real_flow):
    flow_img = cvb.flow2rgb(real_flow)
    bgr_img = cvb.rgb2bgr(flow_img)
    return bgr_img


def showgrayimg(img):
    cv2.imshow(img)
    # plt.show()
    # print img.shape
    return


def save_flow_2_grayimg(dir_flow_path, dir_grayimg_path):
    if not os.path.isdir(dir_flow_path):
        return "not a correct dir"
    for item in os.listdir(dir_flow_path):
        print item
        filename = dir_grayimg_path + item.split('.')[0] + '_img_u.jpg'
        cvb.write_flow(read_flow(dir_flow_path + item), filename, quantize=True)


def save_flow_2_rgbimg(dir_flow_path, dir_rgbimg_path):
    if not os.path.isdir(dir_flow_path):
        return "not a correct dir"
    for item in os.listdir(dir_flow_path):
        print item
        img = flow2rgbimg(read_flow(dir_flow_path + item))
        img = img * 256
     #   cv2.imshow('t1', img)
     #   cv2.waitKey()

     #   print img
     #   rgbArray = np.zeros((240, 320, 3), 'uint8')
     #   rgbArray[..., 0] = img[..., 0] * 256
     #   rgbArray[..., 1] = img[..., 1] * 256
     #   rgbArray[..., 2] = img[..., 2] * 256
     #   img = Image.fromarray(rgbArray)
     #   print img
     #   imshow(img)
     #   show()



     #   print img.shape
    #    cv2.imshow('t1',img)
     #   cv2.waitKey()
        save_img_file_path = dir_rgbimg_path + item.split('.')[0] + '_img.jpg'
        cv2.imwrite(save_img_file_path, img)


def main():
    dir_flow_path = opt.dir_flow_path
#  dir_grayimg_path = opt.dir_grayimg_path
    dir_rgbimg_path = opt.dir_rgbimg_path
#  save_flow_2_grayimg(dir_flow_path, dir_grayimg_path)
    save_flow_2_rgbimg(dir_flow_path, dir_rgbimg_path)

    return

if __name__ == '__main__':
    main()