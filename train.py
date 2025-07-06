import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="1"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!

if __name__ == '__main__':
    model = YOLO('D:\ultralytics-main (1)\ultralytics-main\ultralytics\cfg\models\11\yolo11-HURDNet.yaml')
    # model.load('/home/data/lmy/ultralytics-yolo11-20250415/ultralytics-yolo11-main/yolo11s.pt') # loading pretrain weights
    # model.train(data='/home/data/lmy/UAPD+RDD/UAPD+RDD.yaml',   
    #                    epochs=500,
    #                 #    workers=0,
    #                    project='runs/train',
    #                    device=0,
    #                 #    device=torch.device('cuda:1'),
    #                  #   resume=True,
    #                 #    cache='disk',
    #                    name='yolo11n-UAPD+RDD-HURDNet')

# model.export(format="torchscript")