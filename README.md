# Convolutional Network In Activity Recognition

This project implement some two-stream convolutional network.
> Origin Two-Stream  
> TSN  
> DTPP  

## Data
### UCF101

UCF101 contains 101 actions, 13320 video clips.The dataset can be download here[UCF Dataset](http://crcv.ucf.edu/data/UCF101.php). About 6.93GB.|
### Video -> img

ffmpeg can capture the video's image in one line. Opencv can also do this.
```
    'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
```
details can be find in video_jpg_ucf101_hmdb51.py

### img -> flow

FlowNet2.0 is used here to get the flow.
Use Docker to finish this part. I use the two job  below.  
[NVIDIA-flownet2-python](https://github.com/NVIDIA/flownet2-pytorch)    
[lmb](https://github.com/lmb-freiburg/flownet2-docker)
    

### flow -> img

every flo change into two img, u and v.

## Transfer Learning

## Models

This part include the basebone model in the network.

Four models include here.
> bninception  
> INceptionv4  
> ResNet  
> Inceptionv3  

## caffe to pytorch
### caffe to torch
### torch to pytorch

## spatial_convnet
    
## motion_convnet


## fusion
average_fusion and svm_fusion include here.

## Reference
These module is based on [pytorch](https://github.com/pytorch/pytorch).  
Pretrained module is based on [Cadene](https://github.com/Cadene/pretrained-models.pytorch)  
This origin project is based on jerryhuang's project.
[two-stream-action-recognition](https://github.com/jeffreyhuang1/two-stream-action-recognition) 
