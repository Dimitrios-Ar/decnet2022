import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from decnet import decnet_model
import time
from penet2021.penet2021_model import ENet
import argparse
import os
import numpy as np
sys.path.insert(0, '../utils')
from dataloader import DecnetDataset
from torchvision import transforms
import criteria

from dataset_checker import get_mean_std
from dataset_checker import SanityDatasetCheck
from imutils import paths

from PIL import Image

from custom_metrics import AverageMeter,Result

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="e",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2')
                    #choices=criteria.loss_names,
                    #help='loss function: | '.join(criteria.loss_names) +
                    #' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='/data/dataset/kitti_depth/depth',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default='/data/dataset/kitti_raw',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('--data-folder-save',
                    default='/data/dataset/kitti_depth/submit_test/',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd')
                    #choices=input_options,
                    #help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
#parser.add_argument('--rank-metric',
#                    type=str,
#                    default='rmse',
#                    choices=[m for m in dir(Result()) if not m.startswith('_')],
#                    help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--test', action="store_true", default=False,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

pcl_min = 0 
pcl_max = 10000 #in cm
depth_min = 0
depth_max = 1000 #in cm

training_width = 608
training_height = 352

start = time.time()
dstart = time.time()
'''NEED TO RECALCULATE
mean_rgb = torch.tensor([103.1967,  99.8189,  80.7972])
std_rgb = torch.tensor([51.7836, 47.0790, 40.0790])
mean_d = torch.tensor(116.8378) 
std_d = torch.tensor(162.2317)
mean_gt = torch.tensor(72.2703)
std_gt = torch.tensor(258.6499)

training_width = 608
training_height = 352

def testing(img,mean,std,batch_size,color_type):
    img = img.type(torch.FloatTensor)
    transform_norm = transforms.Normalize((mean,),(std,))
    i=0
    normalized_batch = torch.zeros(batch_size,color_type,352,608)
    for element in img:
        img_normalized = transform_norm(element)
        normalized_batch[i] = img_normalized
        i+=1
    return normalized_batch
'''

def torch_min_max(data):
    minmax = (torch.min(data.float()).item(),torch.max(data.float()).item(),torch.mean(data.float()).item(),torch.median(data.float()).item())
    #print(minmax)
    return minmax

def custom_norm(data,batch_size, d_min, d_max):
    data = data.type(torch.FloatTensor)
    #transform_norm = transforms.Normalize((mean,),(std,))
    i=0
    normalized_batch = torch.zeros(batch_size,1,352,608)
    
    for element in data:
        img_normalized = (element - d_min) / (d_max - d_min)
        normalized_batch[i] = img_normalized
        i+=1
    return normalized_batch

def custom_denorm(data,batch_size, d_min, d_max):
    data = data.type(torch.FloatTensor)
    #transform_norm = transforms.Normalize((mean,),(std,))
    i=0
    normalized_batch = torch.zeros(batch_size,1,352,608)
    
    for element in data:
        img_normalized = element*(d_max-d_min) + d_min
        normalized_batch[i] = img_normalized
        i+=1
    return normalized_batch



#24A
raw_rgb = np.array(Image.open('../../../Desktop/data_sanity/metrics_tester/1638881024930624008_rgb_cropped.png'))
depth = np.array(Image.open('../../../Desktop/data_sanity/metrics_tester/1638881024930624008_depth_cm_cropped.png'))
pcl = np.array(Image.open('../../../Desktop/data_sanity/metrics_tester/1638881024930624008_pcl_cm_cropped.png'))



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((352,608)),
    transforms.ToTensor()
    ])

#print(raw_rgb.shape)
block_average_meter = AverageMeter()
#block_average_meter.reset(False)
average_meter = AverageMeter()
meters = [block_average_meter, average_meter]
print(meters)
for m in meters:
    print(m)

raw_rgb = transform(raw_rgb)
depth = transform(depth)
#print('1',torch_min_max(depth))
depth = custom_norm(depth,1,depth_min,depth_max)
#print('2',torch_min_max(depth))
#depth = custom_denorm(depth,1,depth_min,depth_max)
#print('3',torch_min_max(depth))

#depth = (depth - depth_min) / (depth_max - depth_min)
pcl = transform(pcl)
#pcl = (pcl - pcl_min) / (pcl_max - pcl_min)
pcl = custom_norm(pcl,1,pcl_min,pcl_max)

print(pcl.type)

print(depth.shape)

#depth = depth[None,:,:,:]
#pcl = pcl[None,:,:,:]
rgb = raw_rgb[None,:,:,:]

min_max_rgb = torch_min_max(rgb)
min_max_depth = torch_min_max(depth)
min_max_pcl = torch_min_max(pcl)

#print('RGB: ' + str(min_max_rgb) + '\nDepth: ' + str(min_max_depth) + '\nPcl: ' + str(min_max_pcl))

#depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d,1,color_type=1)
#pcl = testing((torch.from_numpy(np.array(pcl))),mean_gt,std_gt,1,color_type=1)

print(rgb.shape,depth.shape,pcl.shape)
rgb = rgb.to(dtype=torch.float32)

min_max_rgb = torch_min_max(rgb)
min_max_depth = torch_min_max(depth)
min_max_pcl = torch_min_max(pcl)

print('RGB: ' + str(min_max_rgb) + '\nDepth: ' + str(min_max_depth) + '\nPcl: ' + str(min_max_pcl))


model = torch.jit.load('ENETsanity_model_and_weights.pth')



tran = transforms.ToTensor()
new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
        [0.0000, 600.5001220703125, 247.7696533203125],
        [0.0000, 0.0000, 1.0000]])
new_K = tran(new_K)
new_K = new_K.to(dtype=torch.float32)


batch_data = {'rgb': rgb.to(device), 'd': depth.to(device), 'g': pcl.to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}  
with torch.no_grad():
    st1_pred, st2_pred, pred = model(batch_data) 
    depth_criterion = criteria.MaskedMSELoss()
    depth_loss = depth_criterion(pred, pcl.to(device))
    loss = depth_loss
    print(loss)


result = Result()
result.evaluate(pred.data, pcl.to(device).data, photometric=0)
gpu_time = time.time() - start
data_time = time.time() - dstart

print(result.delta1)
m.update(result, gpu_time, data_time, n=1)
#24K
raw_rgb_2 = np.array(Image.open('../../../Desktop/data_sanity/metrics_tester/1636965548558841467_rgb_cropped.png'))
depth_2 = np.array(Image.open('../../../Desktop/data_sanity/metrics_tester/1636965548558841467_depth_cm_cropped.png'))
pcl_2 = np.array(Image.open('../../../Desktop/data_sanity/metrics_tester/1636965548558841467_pcl_cm_cropped.png'))



raw_rgb_2 = transform(raw_rgb_2)
depth_2 = transform(depth_2)
depth_2 = (depth_2 - depth_min) / (depth_max - depth_min)
pcl_2 = transform(pcl_2)
pcl_2 = (pcl_2 - pcl_min) / (pcl_max - pcl_min)


depth_2 = depth_2[None,:,:,:]
pcl_2 = pcl_2[None,:,:,:]
rgb_2 = raw_rgb_2[None,:,:,:]
rgb_2 = rgb_2.to(dtype=torch.float32)

min_max_rgb_2 = torch_min_max(rgb_2)
min_max_depth_2 = torch_min_max(depth_2)
min_max_pcl_2 = torch_min_max(pcl_2)

print('RGB_2: ' + str(min_max_rgb_2) + '\nDepth_2: ' + str(min_max_depth_2) + '\nPcl_2: ' + str(min_max_pcl_2))

batch_data_2 = {'rgb': rgb_2.to(device), 'd': depth_2.to(device), 'g': pcl_2.to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}  
with torch.no_grad():
    t1_pred, st2_pred, pred_2 = model(batch_data_2) 
    depth_criterion = criteria.MaskedMSELoss()
    depth_loss_2 = depth_criterion(pred_2, pcl_2.to(device))
    loss_2 = depth_loss_2
    print(loss_2)


#result_2 = Result()
result.evaluate(pred_2.data, pcl_2.to(device).data, photometric=0)

print(result.delta1)

m.update(result, gpu_time, data_time, n=1)

avg = average_meter.average()

print(avg.delta1)
