#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 18-5-7 下午9:32
     # @Author  : Awiny
     # @Site    :
     # @Project : activity_recognition
     # @File    : resnet101_tsn_spatial.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import spatial_dataloader
from utils.utils import *
from models.ResNet import *
from config import opt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 命令行选项与参数解析
parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default=opt.spatial_checkpoint_path, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')


def main():
    global arg
    #
    arg = parser.parse_args()
    print(arg)

    # Prepare DataLoader
    data_loader = spatial_dataloader.spatial_dataloader(
        BATCH_SIZE=arg.batch_size,  # 批次
        num_workers=8,  # 定义8个子进程加载数据
        path=opt.spatial_train_data_root,
        ucf_list=opt.ucf_list,
        ucf_split=opt.ucf_split,
    )

    train_loader, test_loader, test_video = data_loader.run()
    # test_loader: 71877, 数据加载时的测试集合, 类型： {DataLoader} batch_size:25,
    # train_loader: 9537 类型： {DataLoader}
    # test_video: 类型 {dict} 长度3783， 如： {'Unxxxx_g04_c02' : 96}
    # 得到训练集合等
    # Model
    model = Spatial_CNN(
        nb_epochs=arg.epochs,
        lr=arg.lr,
        batch_size=arg.batch_size,
        resume=arg.resume,
        start_epoch=arg.start_epoch,
        evaluate=arg.evaluate,
        train_loader=train_loader,
        test_loader=test_loader,
        test_video=test_video
    )
    # Training
    model.run()


class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video):
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.test_video = test_video

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        # build model
        self.model = resnet101(pretrained=True, channel=3).cuda()  #
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()  # 损失函数为交叉熵
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=opt.momentum)  # 随机梯度下降，lr为学习系数
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)
        # 降低学习速率，在经过patience次之后，若度量标准没降低，则降低学习速率到'min'

    # resume:使用pre-trained模型
    # 保存和加载整个模型(包括完整神经网络参数)
    # torch.save(model_object, 'model.pth')
    # model = torch.load('model.pth')

    ## 仅保存和加载模型参数
    # torch.save(model_object.state_dict(), 'params.pth')
    # model_object.load_state_dict(torch.load('params.pth'))
    # 从之前运行的结果中继续进行学习
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)  # 从文件加载一个用torch.save保存的对象
                self.start_epoch = checkpoint['epoch']  # 开始次数
                self.best_prec1 = checkpoint['best_prec1']  # 最好预测结果
                self.model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
                self.optimizer.load_state_dict(checkpoint['optimizer'])  # 优化
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                      .format(self.resume, checkpoint['epoch'], self.best_prec1))  #
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        # 评估，直接验证
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()  # 建立模型
        self.resume_and_evaluate()  # 使用pre_trained和进行验证，如果有验证则验证
        cudnn.benchmark = True  # ？

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            acc = self.train_1epoch()  # 训练一个epoch
            prec1, val_loss = self.validate_1epoch()  # 验证预测率和损失函数
            is_best = prec1 > self.best_prec1
            # lr_scheduler
            self.scheduler.step(val_loss)  # 调整学习速率
            # save model，如果是最好模型则保存
            if is_best:
                self.best_prec1 = prec1  # 预测结果
                with open(opt.rgb_preds, 'wb') as f:
                    pickle.dump(self.dic_video_level_preds, f)  # 保存视频级别预测结果
                f.close()

            # 保存训练模型的方法，
            save_checkpoint({
                'epoch': self.epoch,  # 训练的批次
                'state_dict': self.model.state_dict(),  # 参数
                'best_prec1': self.best_prec1,  # 预测记过
                'optimizer': self.optimizer.state_dict()  # ？
            }, is_best, opt.ResNet101_spatial_checkpoint_path, opt.ResNet101_spatial_best_model_path)

    # 训练集，训练1epoch
    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))  # 如70/500，总次数
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        self.model.train()  # 训练
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)  # 进度条
        # get the inputs
        for i, (data_dict, label) in enumerate(progress):
            # measure data loading time
            # 数据加载时间
            data_time.update(time.time() - end)

            label = label.cuda(async=True)  # 标签
            target_var = Variable(label).cuda()  # 目标值

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']), 101).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img' + str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()  # wrap them in Variable
                output += self.model(input_var)  # foraward
            # 计算损失函数
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            #  # zero the parameter gradients
            #  backward + optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()  # 总花费时间

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Data Time': [round(data_time.avg, 3)],
                'Loss': [round(losses.avg, 5)],
                'Prec@1': [round(top1.avg, 4)],
                'Prec@5': [round(top5.avg, 4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, opt.ResNet101_rgb_train_record_path, 'train')
        return top1

    # 验证集
    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)  # tqdm: 进度条，
        for i, (keys, data, label) in enumerate(progress):
            # 得到标签
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            # measure elapsed time ？
            batch_time.update(time.time() - end)
            end = time.time()
            # Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]  # 数目
            for j in range(nb_data):  # 视频名称预测是否正确
                videoName = keys[j].split('/', 1)[0]
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j, :]  # 得到第j个预测的分类结果
                else:
                    self.dic_video_level_preds[videoName] += preds[j, :]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()

        # 保存在csv文件中，loss为numpy格式
        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],  # ？？
                'Loss': [round(video_loss, 5)],
                'Prec@1': [round(video_top1, 3)],
                'Prec@5': [round(video_top5, 3)]}
        record_info(info, opt.ResNet101_rgb_test_record_path, 'test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):

        correct = 0
        # 视频预测结果
        video_level_preds = np.zeros((len(self.dic_video_level_preds), 101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii = 0
        for name in sorted(self.dic_video_level_preds.keys()):

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name]) - 1  # 得到标签对应id

            video_level_preds[ii, :] = preds
            video_level_labels[ii] = label
            ii += 1
            if np.argmax(preds) == (label):  # 预测与id对应相同，正确率累加
                correct += 1

        # top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()  # 得到标签，一维
        video_level_preds = torch.from_numpy(video_level_preds).float()  # 得到预测值，二维

        top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())
        # 计算损失函数

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        # print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1, top5, loss.data.cpu().numpy()


if __name__ == '__main__':
    main()