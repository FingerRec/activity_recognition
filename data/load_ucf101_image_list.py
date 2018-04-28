#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''''''''''''''''''''''''''''''''
     # @Time    : 2018/4/18 10:18
     # @Author  : Awiny
     # @Site    : 
     # @File    : load_ucf101_image_list.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
'''''''''''''''''''''''''''''''''
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

def load_image_list(dir_firstimgs, first_img_list_path, second_img_list_path, optical_flow_path):
    if not os.path.isdir(dir_firstimgs):
        return "not a correct dir"
    local_path = "data/UCF-101-img"
    for dir in os.listdir(dir_firstimgs):
        if dir == ".DS_Store": continue
        if not os.path.isdir(dir_firstimgs + '/'+dir): continue
        for subdir in os.listdir(dir_firstimgs + '/'+dir):
            if subdir == ".DS_Store": continue
            with open(dir_firstimgs + '/' + dir + '/' +subdir + '/n_frames') as f:
                count = int(f.read())
            with open(first_img_list_path, 'at') as f:
                for i in range(1, count):
                    print(i)
                    f.write(local_path + '/' + dir + '/' + subdir + '/' +'image_'+ '{0:05}'.format(i) + '.jpg')
                    f.write("\n")
            with open(second_img_list_path, 'at') as f:
                for i in range(2, count + 1):
                    f.write(local_path + '/' + dir + '/' + subdir + '/' +'image_'+ '{0:05}'.format(i) + '.jpg')
                    f.write("\n")
            with open(optical_flow_path, 'at') as f:
                for i in range(1, count):
                    f.write("data/flow" + '/' + dir + '_' + subdir + '_' + 'image_' + '{0:05}'.format(i) +'_'  + '{0:05}'.format(i+1)+ '.flo')
                    f.write("\n")
            '''
            for img in os.listdir(dir_firstimgs + '/' + dir + '/' +subdir):
                print(img)
                if i != (count - 1) and i < count:
                    with open(first_img_list_path, 'at') as f:
                        f.write(dir_firstimgs + '/' + dir + '/' +subdir + '/' + img)
                        f.write("\n")
                if i != 0 and i < count:
                    with open(second_img_list_path, 'at') as f:
                        f.write(dir_firstimgs + '/' + dir + '/' +subdir + '/' + img)
                        f.write("\n")
                i += 1
            '''
    return

#间隔三帧采样
def load_image_split_list(dir_firstimgs, interval, first_img_split_list_path, second_img_split_list_path, optical_split_flow_path):
    if not os.path.isdir(dir_firstimgs):
        return "not a correct dir"
    local_path = "data/UCF-101-img"
    for dir in os.listdir(dir_firstimgs):
        if dir == ".DS_Store": continue
        if not os.path.isdir(dir_firstimgs + '/'+dir): continue
        for subdir in os.listdir(dir_firstimgs + '/'+dir):
            if subdir == ".DS_Store": continue
            with open(dir_firstimgs + '/' + dir + '/' +subdir + '/n_frames') as f:
                count = int(f.read())
            with open(first_img_split_list_path, 'at') as f:
                for i in range(1, count):
                    if i % interval == 0:
                        f.write(local_path + '/' + dir + '/' + subdir + '/' +'image_'+ '{0:05}'.format(i) + '.jpg')
                        f.write("\n")
            with open(second_img_split_list_path, 'at') as f:
                for i in range(interval, count + 1):
                    if (i - interval + 1) % interval == 0:
                        f.write(local_path + '/' + dir + '/' + subdir + '/' +'image_'+ '{0:05}'.format(i) + '.jpg')
                        f.write("\n")
            with open(optical_split_flow_path, 'at') as f:
                for i in range(interval, count + 1):
                    if (i - interval + 1) % interval == 0:
                        print(i)
                        f.write("data/flow" + '/' + dir + '_' + subdir + '_' + 'image_' + '{0:05}'.format(i-interval+1) +'_'  + '{0:05}'.format(i)+ '.flo')
                        f.write("\n")
            '''
            for img in os.listdir(dir_firstimgs + '/' + dir + '/' +subdir):
                print(img)
                if i != (count - 1) and i < count:
                    with open(first_img_list_path, 'at') as f:
                        f.write(dir_firstimgs + '/' + dir + '/' +subdir + '/' + img)
                        f.write("\n")
                if i != 0 and i < count:
                    with open(second_img_list_path, 'at') as f:
                        f.write(dir_firstimgs + '/' + dir + '/' +subdir + '/' + img)
                        f.write("\n")
                i += 1
            '''
    return

if __name__ == "__main__":
    '''
    dir_imgs_path = "/数据库/UCf-101-jpg"
    first_img_list_path = "/数据库/UCf-101-jpg/first_img_list.txt"
    second_img_list_path = "/数据库/UCf-101-jpg/second_img_list.txt"
    optical_flow_path = "/数据库/UCf-101-jpg/optical_flow_list.txt"
    load_image_list(dir_imgs_path, first_img_list_path, second_img_list_path, optical_flow_path)
    '''
    interval = 5
    dir_imgs_path = "/数据库/UCf-101-jpg"
    first_img_split_list_path = "/数据库/UCf-101-jpg/first_img_split_list.txt"
    second_img_split_list_path = "/数据库/UCf-101-jpg/second_img_split_list.txt"
    optical_flow_split_path = "/数据库/UCf-101-jpg/optical_flow_split_list.txt"
    load_image_split_list(dir_imgs_path,interval, first_img_split_list_path, second_img_split_list_path, optical_flow_split_path)