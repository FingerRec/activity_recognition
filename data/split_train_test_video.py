# -*- coding: utf-8 -*-
import os, pickle
from config import opt

class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    #get anction name and id,save in action_label
    #classInd.txt
    def get_action_index(self):
        self.action_label={}
        with open(opt.ucf_list + 'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')
            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label

    #split video,
    #path:UCF_list
    def split_video(self):
        self.get_action_index() # get action label
        for path,subdir,files in os.walk(self.path): #get path,subdir,files
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename) #get trainlist01.txt
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename) #get testlist01.txt
        print '==> (Training video, Validation video):(', len(train_video),len(test_video),')'
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    #open trainlist01.txt and extract video name
    #从txt里读入，变成字典格式， 对应avi名称和标签
    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content] #去掉换行/回撤
        f.close()
        dic={}
        for line in content:
            #print line
            video = line.split('/',1)[1].split(' ',1)[0] #  video:v_ApplyEyeMakeup_g08_c01.avi
            key = video.split('_',1)[1].split('.',1)[0] #v_ApplyEyeMakeup_g08_c01
            label = self.action_label[line.split('/')[0]]   # "1"
            dic[key] = int(label)
            #print key,label
        return dic #dictonary

    def name_HandstandPushups(self,dic): #？just for handstandpushups + '_'
        dic2 = {}
        for video in dic:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            else:
                videoname = video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':

    path = '../UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    #train_video : 从trainlist01 读取的视频名和标签的对应字典，长度9537
    #分为测试集和训练集，均为保存的文件名，没有文件实体
    #根据切分，读入trainlist01和testlist01
    print len(train_video),len(test_video)