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
from dataset_checker  import visualize_batch
from imutils import paths

import wandb
import vis_utils



from custom_metrics import AverageMeter,Result


def torch_min_max(data):
    minmax = (torch.min(data.float()).item(),torch.max(data.float()).item(),torch.mean(data.float()).item(),torch.median(data.float()).item())
    #print(minmax)
    return minmax


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


def testing(img,mean,std,batch_size,color_type):
    img = img.type(torch.FloatTensor)
    transform_norm = transforms.Normalize((mean),(std))
    i=0
    normalized_batch = torch.zeros(batch_size,color_type,training_height,training_width)
    #print('normalized_batch_sze', normalized_batch.shape)
    for element in img:
        img_normalized = transform_norm(element)
        #print('img_normalizedshape', img_normalized.shape)
        normalized_batch[i] = img_normalized
        i+=1
    return normalized_batch

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

training_width = 608
training_height = 352
args.val_h = training_height#352
args.val_w = training_width#1216
train_continue = True

evaluation = True
precalculated_mean_std = True
random_seed = 2910
pcl_min = 0 
pcl_max = 10000 #in cm
depth_min = 0
depth_max = 1000 #in cm
best_prev_rmse = np.inf
lowest_loss = np.inf
block_average_meter = AverageMeter()
#block_average_meter.reset(False)
average_meter = AverageMeter()
meters = [block_average_meter, average_meter]
result = Result()
for m in meters:
    pass


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

batch_size = 1
train_folder_loc = Path('../../../Desktop/data_sanity/24k_cropped_reduced')
#train_folder_loc = Path('../../../Desktop/data_sanity/testbatch')

test_folder_loc = Path('../../../Desktop/data_sanity/24a_cropped_reduced')

crop_transform = transforms.CenterCrop(352)

rgbPath = np.array(sorted(list(paths.list_images(os.path.join(train_folder_loc,'rgb_cropped')))))
depthPath = np.array(sorted(list(paths.list_images(os.path.join(train_folder_loc,'depth_cm_cropped')))))
pclPath = np.array(sorted(list(paths.list_images(os.path.join(train_folder_loc,'pcl_cm_cropped')))))

train_mask = np.random.choice(len(rgbPath), 10, replace=False)

train_mask = train_mask.astype(int)
#print(type(train_mask))
mini_train_set = SanityDatasetCheck(rgbPath[train_mask],depthPath[train_mask],pclPath[train_mask])

#ransformed_mini_train_set = crop_transform(mini_train_set)

rgbPath = np.array(sorted(list(paths.list_images(os.path.join(test_folder_loc,'rgb_cropped')))))
depthPath = np.array(sorted(list(paths.list_images(os.path.join(test_folder_loc,'depth_cm_cropped')))))
pclPath = np.array(sorted(list(paths.list_images(os.path.join(test_folder_loc,'pcl_cm_cropped')))))

test_mask = np.random.choice(len(rgbPath), 500, replace=False)
mini_test_set = SanityDatasetCheck(rgbPath[test_mask],depthPath[test_mask],pclPath[test_mask])

#transformed_mini_test_set = crop_transform(mini_test_set)

print(len(mini_train_set),len(mini_test_set))
train_dl = DataLoader(mini_train_set, batch_size=batch_size)
test_dl = DataLoader(mini_test_set,batch_size=1)
print(test_dl)

'''
rgbPath = np.array(list(paths.list_images(os.path.join(train_folder_loc,'rgb_cropped'))))
depthPath = np.array(list(paths.list_images(os.path.join(train_folder_loc,'depth_cm_cropped'))))
pclPath = np.array(list(paths.list_images(os.path.join(train_folder_loc,'pcl_cm_cropped'))))
mini_train_set = SanityDatasetCheck(rgbPath,depthPath,pclPath)
train_dl = DataLoader(mini_train_set, batch_size=batch_size)
'''
'''
test_folder_loc = Path('../../../Desktop/data_sanity/nn_dataset_24a_cropped')
rgbPath_test = np.array(list(paths.list_images(os.path.join(test_folder_loc,'rgb_cropped'))))
depthPath_test = np.array(list(paths.list_images(os.path.join(test_folder_loc,'depth_cm_cropped'))))
pclPath_test = np.array(list(paths.list_images(os.path.join(test_folder_loc,'pcl_cm_cropped'))))

test_mask = np.random.choice(len(rgbPath_test), 500, replace=False)
mini_test_set = SanityDatasetCheck(rgbPath_test[test_mask],depthPath_test[test_mask],pclPath_test[test_mask])
#mini_test_set = SanityDatasetCheck(rgbPath,depthPath,pclPath)
'''
#train_dl = DataLoader(mini_train_set, batch_size=batch_size)
#test_dl = DataLoader(mini_test_set,batch_size=1)

if precalculated_mean_std == False:
    print("Calculating training dataset mean,std")
    mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt = get_mean_std(train_dl)
    print(mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt)

epochs = 100
lr = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

submodel = 'ENETsanity'
model = ENet(args).to(device)
if train_continue == True:

    model = torch.jit.load('ENETsanity_model_and_weights_TRAIN.pth')


model_save_name_eval = submodel+'_model_and_weights_EVAL_2070.pth'
model_save_name_train = submodel+'_model_and_weights_TRAIN_2070.pth'

print(model_save_name_eval)
wandblogger = False
if wandblogger == True:
        wandb.init(project="decnet-project", entity="wandbdimar")
        wandb.config = {
            "model": submodel,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size
            }

#model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((352,608)),
    transforms.ToTensor()
    ])


best_loss_and_epoch = float("inf")
training_start_time = time.time()
if precalculated_mean_std == True:
    mean_rgb = torch.tensor([103.1967,  99.8189,  80.7972])
    std_rgb = torch.tensor([51.7836, 47.0790, 40.0790])
    mean_d = torch.tensor(116.8378) 
    std_d = torch.tensor(162.2317)
    mean_gt = torch.tensor(72.2703)
    std_gt = torch.tensor(258.6499)

for epoch in range(1,epochs+1):#how many epochs to run
    if epoch % 100 == 0:
        print("new lr")
        lr = lr/2    
    epoch_start_time = time.time()
    epoch_iter = 0
    sum_loss = 0
    average_loss = float("inf")
    
    for i, data in enumerate(train_dl,start=epoch_iter):
        if evaluation == True:
            break
        optimizer.zero_grad()
        rgb, depth, pcl = data[0], data[1], data[2]
        depth = custom_norm(depth,batch_size,depth_min,depth_max)

        pcl = custom_norm(pcl,batch_size,pcl_min,pcl_max)

        min_max_rgb = torch_min_max(rgb)
        min_max_depth = torch_min_max(depth)
        min_max_pcl = torch_min_max(pcl)


        rgb = rgb.to(dtype=torch.float32)

        epoch_iter += batch_size
        tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
        new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                [0.0000, 600.5001220703125, 247.7696533203125],
                [0.0000, 0.0000, 1.0000]])
        new_K = tran(new_K)
        new_K = new_K.to(dtype=torch.float32)

        batch_data = {'rgb': rgb.to(device), 'd': depth.to(device), 'g': pcl.to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}  
        st1_pred, st2_pred, pred = model(batch_data) 
        depth_criterion = criteria.MaskedMSELoss()
        depth_loss = depth_criterion(pred, pcl.to(device))

        #print('pred_data', torch_min_max(pred))
        loss = depth_loss
        loss.backward()
        optimizer.step()

        sum_loss += loss
        average_loss = sum_loss / (epoch_iter / batch_size)
        print('\n', 'Average loss: ', average_loss.item(), ' --- Iterations: ' ,epoch_iter, ' --- Epochs: ', epoch, '\n')
        if average_loss < lowest_loss: 
            save_batch = {'rgb': torch.ones(1,3,training_height,training_width).to(device), 'd': torch.ones(1,1,training_height,training_width).to(device), 'g': torch.ones(1,1,training_height,training_width).to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}
            trace_model = torch.jit.trace(model,save_batch)        
            torch.jit.save(trace_model, model_save_name_train)
            lowest_loss = average_loss
    
    with torch.no_grad():
        print('\nStarting evaluation')
        dstart = time.time()
        #avg = None
        m.reset()
        for i_eval, data_eval in enumerate(train_dl):
            #visualize_batch(data_eval)
            str_i = str(epoch+1)
            path_i = 'epoch_' + str_i.zfill(4) + '.png'
            path_rgb = os.path.join('test_data/rgb', path_i)
            path_pcl = os.path.join('test_data/pcl', path_i)
            path_depth = os.path.join('test_data/depth', path_i)
            path_pred = os.path.join('test_data/pred', path_i)
            #print(path_pred)
            start = time.time()
            rgb, depth, pcl = data_eval[0], data_eval[1], data_eval[2]
            #vis_utils.save_depth_as_uint16png_upload(pcl, path_pcl)
            #vis_utils.save_depth_as_uint16png_upload(depth, path_depth)
            #vis_utils.save_image_torch(rgb,path_rgb)
            min_max_rgb = torch_min_max(rgb)
            min_max_depth = torch_min_max(depth)
            rgb = rgb.to(dtype=torch.float32)

            epoch_iter += batch_size
            tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
            new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                    [0.0000, 600.5001220703125, 247.7696533203125],
                    [0.0000, 0.0000, 1.0000]])
            new_K = tran(new_K)
            new_K = new_K.to(dtype=torch.float32)

            batch_data = {'rgb': rgb.to(device), 'd': depth.to(device), 'g': pcl.to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}  
            st1_pred, st2_pred, pred = model(batch_data) 
            depth_criterion = criteria.MaskedMSELoss()
            depth_loss = depth_criterion(pred, pcl.to(device))

            loss = depth_loss
            pred = custom_denorm(pred,1,pcl_min,pcl_max)
            pcl = custom_denorm(pcl,1,pcl_min,pcl_max)
            depth = custom_denorm(depth,1,depth_min,depth_max)

            
            min_max_normalized_pcl = torch_min_max(pcl)
            min_max_normalized_pred = torch_min_max(pred)

            #print('pcl_pred_after',min_max_normalized_pcl,min_max_normalized_pred)
            result.evaluate(pred.data, pcl.data, photometric=0)
            #vis_utils.save_depth_as_uint16png_upload(pred, path_pred)


            gpu_time = time.time() - start
            data_time = time.time() - dstart
            m.update(result, gpu_time, data_time, n=1)
            avg = average_meter.average()

            progress = 100*(i_eval/len(mini_test_set))

        print('Average eval rmse: ' + str(avg.rmse) + ' in ' + str(data_time) + ' seconds')
        vis_utils.save_depth_as_uint16png_upload(pcl, path_pcl)
        vis_utils.save_depth_as_uint16png_upload(depth, path_depth)
        vis_utils.save_depth_as_uint16png_upload(pred, path_pred)
        vis_utils.save_image_torch(rgb,path_rgb)
        
        if avg.rmse < best_prev_rmse:
            best_prev_rmse = avg.rmse
            print('best_prev_rmse', best_prev_rmse)
            print(type(best_prev_rmse))
            save_batch = {'rgb': torch.ones(1,3,training_height,training_width).to(device), 'd': torch.ones(1,1,training_height,training_width).to(device), 'g': torch.ones(1,1,training_height,training_width).to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}
            trace_model = torch.jit.trace(model,save_batch)        
            torch.jit.save(trace_model, model_save_name_eval)
            
        if wandblogger == True:
            wandb.log({"average batch loss": average_loss,
            "best previous evaluaion rmse": avg.rmse})

'''
    if average_loss < best_loss_and_epoch:

        best_loss_and_epoch = average_loss
        #torch.save(model, 'model_test_best_nn.pth')
        #print('saving model at ',average_loss.item,epoch_iter,epoch)
        print('saving model and weights at ',average_loss.item,epoch_iter,epoch)
        #save_batch = (torch.ones(1,3,training_height,training_width).to(device), torch.ones(1,1,training_height,training_width).to(device))
        save_batch = {'rgb': torch.ones(1,3,training_height,training_width).to(device), 'd': torch.ones(1,1,training_height,training_width).to(device), 'g': torch.ones(1,1,training_height,training_width).to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}
        
        with torch.no_grad():
            trace_model = torch.jit.trace(model,save_batch)
        
        torch.jit.save(trace_model, model_save_name)
    training_duration = training_start_time - time.time()
    #print('Epoch training duration: ', training_duration)
print('Total training duration: ', training_duration)

if evaluation == True:
    model = torch.jit.load('best_ENETsanity_model_and_weights.pth')
    #model = torch.jit.load('decnet_model_and_weights.pth')

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        print('\nStarting evaluation')
        for i_eval, data_eval in enumerate(test_dl):
            rgb, depth, pcl = data_eval[0], data_eval[1], data_eval[2]
            rgb = torch.transpose(rgb, 3,1)
            rgb = torch.transpose(rgb, 3,2)
            #print(rgb_float.shape)
            depth = depth[:,:,:,None]
            depth = torch.transpose(depth, 3,1)
            depth = torch.transpose(depth, 3,2)

            pcl = pcl[:,:,:,None]
            pcl = torch.transpose(pcl, 3,1)
            pcl = torch.transpose(pcl, 3,2)

            #rgb = testing((torch.from_numpy(np.array(rgb))),mean_rgb,std_rgb,batch_size,color_type=3)       
            depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d,1,color_type=1)
            pcl = testing((torch.from_numpy(np.array(pcl))),mean_gt,std_gt,1,color_type=1)

            rgb = torch.transpose(rgb, 2,1)
            rgb = torch.transpose(rgb, 3,2)
            
            rgb = rgb.to(dtype=torch.float32)

            epoch_iter += 1
            #print("MIN_image_bf",torch.min(image.float()),"MEAN_image_bf", torch.mean(image.float()),"MEDIAN_image_bf", torch.median(image.float()),"MAX_image_bf",torch.max(image.float()))
            #print("MIN_depth_bf",torch.min(depth.float()).item(),"MEAN_depth_bf", torch.mean(depth.float()).item(),"MEDIAN_depth_bf", torch.median(depth.float()).item(),"MAX_depth_bf",torch.max(depth.float()).item())
            tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
            #output = model(image.to(device),depth.to(device))
            new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                    [0.0000, 600.5001220703125, 247.7696533203125],
                    [0.0000, 0.0000, 1.0000]])
            new_K = tran(new_K)
            new_K = new_K.to(dtype=torch.float32)
            batch_data = {'rgb': rgb.to(device), 'd': depth.to(device), 'g': pcl.to(device), 'position': torch.zeros(1, 3, training_height, training_width).to(device), 'K': new_K.to(device)}  
            #pred = model(rgb.to(device),depth.to(device))
            #pred = model(batch_data)
            st1_pred, st2_pred, pred = model(batch_data) 
            #output_loss = pred
            depth_criterion = criteria.MaskedMSELoss()
            depth_loss = depth_criterion(pred, pcl.to(device))


            loss = depth_loss
            
            sum_loss += loss
            average_loss = sum_loss / (epoch_iter / batch_size)
            progress = 100*epoch_iter/len(mini_test_set)
            print('\n', 'Average loss: ', average_loss.item(), ' --- Progress: ' , progress , '%\n')
'''
        