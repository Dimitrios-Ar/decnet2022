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



def testing(img,mean,std):
    img = img.type(torch.FloatTensor)
    #print(img)
    #print(mean,std)
    transform_norm = transforms.Normalize((mean,),(std,))
    #print(img.dtype)
    i=0
    normalized_batch = torch.zeros(8,1,368,640)
    for element in img:

        
        #print(element.shape)
        img_normalized = transform_norm(element)
        normalized_batch[i] = img_normalized
        i+=1
    return normalized_batch

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
args.val_h = 368#352
args.val_w = 640#1216


base_path = Path('../../data/dataset_nn/nn_24k_cropped_640_368')

data = DecnetDataset(base_path/'rgb_cropped',
                     base_path/'depth_cm_cropped',
                     base_path/'pcl_cm_cropped')

epochs = 1000
lr = 1e-3
#split_ratio = 0.886
split_ratio = 0.0095

batch_size = 8

testing_image = int(len(data)*split_ratio)


train_ds, test_ds = torch.utils.data.random_split(data, ((len(data)-int(len(data)*split_ratio), int(len(data)*split_ratio))))

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

print(len(train_ds))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


submodel = 'DecNetRGBDsmall'
model = decnet_model.DecNetRGBDsmall().to(device)

model = ENet(args).to(device)

#odel = torch.load('../weights/model_test_best_nn.pth', map_location='cuda')
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#print(model)

#xb, yb = next(iter(train_dl))

criterion = torch.nn.MSELoss()

best_loss_and_epoch = 1000000.0
training_start_time = time.time()

mean_rgb = torch.tensor([103.1967,  99.8189,  80.7972])
std_rgb = torch.tensor([51.7836, 47.0790, 40.0790])
mean_d = torch.tensor(116.8378)
std_d = torch.tensor(162.2317)
mean_gt = torch.tensor(72.2703)
std_gt = torch.tensor(258.6499)

wandblogger = False
if wandblogger == True:
        wandb.init(project="decnet-project", entity="wandbdimar")
        wandb.config = {
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 8
            }

for epoch in range(1,epochs+1):#how many epochs to run
    epoch_start_time = time.time()
    epoch_iter = 0
    sum_loss = 0
    for i, data in enumerate(train_dl,start=epoch_iter):

        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2, round3 = 1, 3, None
        

        image, depth, gt = data[0][:,0:3,:,:], data[0][:,3:,:,:], data[1]
        #print(depth)
        depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d)
        gt = testing((torch.from_numpy(np.array(gt))),mean_gt,std_gt)
        #print(depth.size(),gt.size())
        #transform_norm = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.Normalize(mean_d, mean_d)
        #])

        #depth = transform_norm(depth)
        #print(image.shape,depth.shape,gt.shape)
        epoch_iter += batch_size
        #print("MIN_image_bf",torch.min(image.float()),"MEAN_image_bf", torch.mean(image.float()),"MEDIAN_image_bf", torch.median(image.float()),"MAX_image_bf",torch.max(image.float()))
        #print("MIN_depth_bf",torch.min(depth.float()).item(),"MEAN_depth_bf", torch.mean(depth.float()).item(),"MEDIAN_depth_bf", torch.median(depth.float()).item(),"MAX_depth_bf",torch.max(depth.float()).item())
        tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
        #output = model(image.to(device),depth.to(device))
        new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                [0.0000, 600.5001220703125, 247.7696533203125],
                [0.0000, 0.0000, 1.0000]])
        new_K = tran(new_K)
        new_K = new_K.to(dtype=torch.float32)
        #print(new_K.shape)
        #print(image.shape,depth.shape)
        batch_data = {'rgb': image.to(device), 'd': depth.to(device), 'g': data[1].to(device), 'position': torch.zeros(1, 3, 368, 640).to(device), 'K': new_K.to(device)}  
        st1_pred, st2_pred, pred = model(batch_data) 
        #pred = model(image.to(device),depth.to(device))
        output_loss = pred
        #print(pred)
        #gt_loss = gt.to(device).to(torch.float32)
        #output_loss = output.to(torch.float32)
        #loss = criterion(gt.to(device),output.to(device))
        
        #print("MIN_gt",torch.min(gt.float()).item(),"MEAN_gt", torch.mean(gt.float()).item(),"MEDIAN_gt", torch.median(gt.float()).item(),"MAX_gt",torch.max(gt.float()).item())
        #print("MIN_gt_loss",torch.min(gt_loss.float()),"MEAN_gt_loss", torch.mean(gt_loss.float()),"MEDIAN_gt_loss", torch.median(gt_loss.float()),"MAX_gt_loss",torch.max(gt_loss.float()))
        #print("MIN_output_loss",torch.min(output_loss.float()).item(),"MEAN_output_loss", torch.mean(output_loss.float()).item(),"MEDIAN_output_loss", torch.median(output_loss.float()).item(),"MAX_output_loss",torch.max(output_loss.float()).item())
        depth_criterion = criteria.MaskedMSELoss()
        depth_loss = depth_criterion(pred, gt.to(device))

        #st1_loss = depth_criterion(st1_pred, gt.to(device))
        #st2_loss = depth_criterion(st2_pred, gt.to(device))
        #print(st1_loss,st2_loss)
        #loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss
        loss = depth_loss
        loss.backward()
        #print(pred.shape)
        #print(len(torch.nonzero(pred)))
        #print(len(torch.nonzero(gt.to(device))))
        
        sum_loss += loss
        optimizer.step()
        average_loss = sum_loss / (epoch_iter / batch_size)
        print('\n', 'Average loss: ', average_loss.item(), ' --- Iterations: ' ,epoch_iter, ' --- Epochs: ', epoch, '\n')
    if average_loss < best_loss_and_epoch:

        best_loss_and_epoch = average_loss
        torch.save(model, 'model_test_best_nn.pth')
        print('saving model at ',average_loss.item,epoch_iter,epoch)
        print('saving model and weights at ',average_loss.item,epoch_iter,epoch)
        save_batch = {'rgb': torch.ones(1,3,368,640).to(device), 'd': torch.ones(1,1,368,640).to(device), 'g': torch.ones(1,1,368,640).to(device), 'position': torch.zeros(1, 3, 368, 640).to(device), 'K': new_K.to(device)}
        with torch.no_grad():
            trace_model = torch.jit.trace(model,save_batch)
        torch.jit.save(trace_model,'decnet_model_and_weights.pth')

    if wandblogger == True:
        wandb.log({"average batch loss": average_loss})

        # Optional
        wandb.watch(model)