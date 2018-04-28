# -*- coding: utf-8 -*-
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure

class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)
    #每张图进行预处理操作，并返回，处理的是该视频的第index张图片
    def load_ucf_image(self,video_name, index):
        '''
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandStandPushups_'+g
       #     path = self.root_dir + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
            path = self.root_dir + 'v_' + name + '/frame'
        else:
        '''
       # path = self.root_dir + video_name.split('_')[0]+'/separated_images/v_'+video_name+'/v_'+video_name+'_'
        path = self.root_dir +  'v_' + video_name + '/frame'
        img = Image.open(path + '{0:06}'.format(index)+'.jpg') #342x256,RGB图像
        transformed_img = self.transform(img) #变为3x224x224
        img.close()

        return transformed_img
    #__getitem__???
    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')#{Swing_g09_c02, 116}
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, nb_clips/3))
            clips.append(random.randint(nb_clips/3, nb_clips*2/3))
            clips.append(random.randint(nb_clips*2/3, nb_clips+1))
            #clips = [23,67,93] 随机选取三张图片
        elif self.mode == 'val':
            #验证，只选取一张图片
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx] #88
        label = int(label)-1 #label-1:对应真实类别
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index) #裁剪并返回，img1,img2,img3
                    
            sample = (data, label) #tuple,包含三张图片？？，为3x224x224; 字典和标签。
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    #BATCH：1， num_workers:1
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video() #返回两个字典

    #从frame_count.pickle 中加载每个视频对应的帧数
    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open(opt.frame_count, 'rb') as file:
            dic_frame = pickle.load(file) #字典，对应视频名称 {xxx.avi, num}, 长度13320，这里的帧图像个数和我用的数据库有一点点出入
        file.close()

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0] #'Lunges_g07_c01'
            n,g = videoname.split('_',1) #n:Lunges,g:g07_c01,
            if n == 'HandStandPushups': #?????
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line] #{dict} {'Lunges_g07_c01':248(帧数)}

    def run(self):
        self.load_frame_count() #得到帧数 {'Lunges_g07_c01':248(帧数)}
        self.get_training_dic() #得到训练集字典 {'Milsxxxx_g17_c04 133':53}
        self.val_sample20() #得到测试集字典，每19张图片采样一张
        train_loader = self.train()
        # DataLoader格式，keys: {list 9537} {'Vxx_g20_c04 47'},value: 9537,每个对应类别（1-101），transform为四种格式
        val_loader = self.validate()
        #验证集
        return train_loader, val_loader, self.test_video
    #
    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}
        for video in self.train_video:
            #print videoname
            nb_frame = self.frame_count[video]-10+1 #nb_frame为目录下所有帧-10+1 ？
            key = video+' '+ str(nb_frame) #得到key:
            self.dic_training[key] = self.train_video[video] #{'Milsxxxx_g17_c04 133':53}

    #得到测试字典
    def val_sample20(self):
        print '==> sampling testing frames'
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19) #
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]       #{'pxxxx_g01_c03 1'} = 69  1 20  39 ....

    #组合transforms,随机裁剪至224，RandomHorizontalFlip 随机水平翻转一半，img转换为tensor
    #Normalize:给定均值：(R,G,B) 方差：（R，G，B）,将会把Tensor正则化。即：Normalized_image=(image-mean)/std。
    def train(self):
        #得到训练数据集，？？？
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        #9537，数目竟然没变？
        print '==> Training data :',len(training_set),'frames'
        print training_set[1][0]['img1'].size()
        #利用pyTorch返回随机数据集
        train_loader = DataLoader(
            dataset=training_set,  #加载数据的数据集
            batch_size=self.BATCH_SIZE, #每个batch加载多少个样本
            shuffle=True, #每个epoch重新打乱数据
            num_workers=self.num_workers) #加载数据的进程个数
        #9537张图？
        return train_loader #DataLoader格式，keys: {list 9537} {'Vxx_g20_c04 47'},value: 9537,每个对应类别（1-101），transform为四种格式

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print '==> Validation data :',len(validation_set),'frames'
        print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader





if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/数据库/UCF-101切片/jpegs_256/',
                                ucf_list='../UCF_list/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()