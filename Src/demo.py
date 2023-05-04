import torch
import torch.nn as nn
import torchvision.models as models
from Src.backbone.ResNet import ResNet_2Branch

resnet50 = models.resnet50(pretrained=True)
        #带参数的模型
#print(resnet50)
pretrained_dict = resnet50.state_dict()
        #加载模型参数
#print(pretrained_dict)
all_params = {}
        #创建一个空的字典#
for k, v in ResNet_2Branch().state_dict().items():
    #print('k',k)
    #print(v)
    #k指的是在哪一层
    #指的是该层的参数
    if k in pretrained_dict.keys():
        v = pretrained_dict[k]
        all_params[k] = v
    elif '_1' in k:
        name = k.split('_1')[0] + k.split('_1')[1]
        v = pretrained_dict[name]
        all_params[k] = v
    elif '_2' in k:
        name = k.split('_2')[0] + k.split('_2')[1]
        v = pretrained_dict[name]
        all_params[k] = v
    #assert len(all_params.keys()) == len(ResNet_2Branch().state_dict().keys())