import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.LINet import LINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import time
from numpy import mean
torch.cuda.synchronize()
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
#输入图像的尺寸
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2023-LINet/LINet_40.pth')
#模型存放位置
parser.add_argument('--test_save', type=str,
                    default='./Result/2023-LINet-New/')
#测试结果保存位置
opt = parser.parse_args()

model = LINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
                                 #获取模型参数
                      #加载模型文件
#torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
model.eval()
#评估模式

for dataset in ['COD10K']:
#for dataset in ['CAMO']:
#for dataset in ['CHAMELEON']:
    save_path = opt.test_save + dataset + '/'
    #文件保存路径
    os.makedirs(save_path, exist_ok=True)
    #递归创建目录，创建文件夹。exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。

    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    #加载测试数据
    img_count = 1
    start = time.time()
    for iteration in range(test_loader.size):
        # load data
        time_list = []
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        #作用是将输入转换为数组
        gt /= (gt.max() + 1e-8)
        #归一化
        image = image.cuda()
        #gpu运算
        # inference
        start_each = time.time()
        cam= model(image)
        time_each = time.time() - start_each
        time_list.append(time_each)

        # reshape and squeeze
        cam = F.interpolate (cam, size=gt.shape, mode='bilinear', align_corners=True)

        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1
print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))
end = time.time()
print('程序执行时间: ',end-start)
print("\n[Congratulations! Testing Done]")
torch.cuda.synchronize() #增加同步操作
