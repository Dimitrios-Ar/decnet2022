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



def testing(img,mean,std,batch_size,color_type):
    img = img.type(torch.FloatTensor)
    transform_norm = transforms.Normalize((mean,),(std,))
    i=0
    normalized_batch = torch.zeros(batch_size,color_type,360,640)
    for element in img:
        img_normalized = transform_norm(element)
        normalized_batch[i] = img_normalized
        i+=1
    return normalized_batch


evaluation = False
precalculated_mean_std = True
random_seed = 2910


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

batch_size = 8
train_folder_loc = Path('../../../Desktop/data_sanity/24k_cropped_reduced')
#train_folder_loc = Path('../../../Desktop/data_sanity/testbatch')

test_folder_loc = Path('../../../Desktop/data_sanity/24a_cropped_reduced')



rgbPath = np.array(list(paths.list_images(os.path.join(train_folder_loc,'rgb_cropped'))))
depthPath = np.array(list(paths.list_images(os.path.join(train_folder_loc,'depth_cm_cropped'))))
pclPath = np.array(list(paths.list_images(os.path.join(train_folder_loc,'pcl_cm_cropped'))))

train_mask = np.random.choice(len(rgbPath), 2000, replace=False)

train_mask = train_mask.astype(int)
#print(type(train_mask))
mini_train_set = SanityDatasetCheck(rgbPath[train_mask],depthPath[train_mask],pclPath[train_mask])

rgbPath = np.array(list(paths.list_images(os.path.join(test_folder_loc,'rgb_cropped'))))
depthPath = np.array(list(paths.list_images(os.path.join(test_folder_loc,'depth_cm_cropped'))))
pclPath = np.array(list(paths.list_images(os.path.join(test_folder_loc,'pcl_cm_cropped'))))

test_mask = np.random.choice(len(rgbPath), 500, replace=False)
mini_test_set = SanityDatasetCheck(rgbPath[test_mask],depthPath[test_mask],pclPath[test_mask])

print(len(mini_train_set))
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

submodel = 'DecNetRGBDsmall'
model = decnet_model.DecNetRGBDsmall().to(device)



#model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


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
        #print(data[3])
        rgb, depth, pcl = data[0], data[1], data[2]
        rgb = torch.transpose(rgb, 3,1)
        rgb = torch.transpose(rgb, 3,2)
        #print(rgb_float.shape)
        depth = depth[:,:,:,None]
        depth = torch.transpose(depth, 3,1)
        depth = torch.transpose(depth, 3,2)

        pcl = pcl[:,:,:,None]
        pcl = torch.transpose(pcl, 3,1)
        pcl = torch.transpose(pcl, 3,2)

        #print("MIN_depth_bf",torch.min(depth.float()),"MEAN_image_bf", torch.mean(depth.float()),"MEDIAN_image_bf", torch.median(depth.float()),"MAX_image_bf",torch.max(depth.float()))
        #print("MIN_pcl_bf",torch.min(pcl.float()),"MEAN_image_bf", torch.mean(pcl.float()),"MEDIAN_image_bf", torch.median(pcl.float()),"MAX_image_bf",torch.max(pcl.float()))

        #rgb = testing((torch.from_numpy(np.array(rgb))),mean_rgb,std_rgb,batch_size,color_type=3)       
        depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d,batch_size,color_type=1)
        pcl = testing((torch.from_numpy(np.array(pcl))),mean_gt,std_gt,batch_size,color_type=1)
        #print("MIN_depth_normalize_bf",torch.min(depth.float()),"MEAN_image_bf", torch.mean(depth.float()),"MEDIAN_image_bf", torch.median(depth.float()),"MAX_image_bf",torch.max(depth.float()))
        #print("MIN_pcl_normalize_bf",torch.min(pcl.float()),"MEAN_image_bf", torch.mean(pcl.float()),"MEDIAN_image_bf", torch.median(pcl.float()),"MAX_image_bf",torch.max(pcl.float()))
        
        depth = (torch.from_numpy(np.array(depth)).type(torch.FloatTensor))
        pcl = (torch.from_numpy(np.array(pcl)).type(torch.FloatTensor))
        
        
        rgb = rgb.to(dtype=torch.float32)

        epoch_iter += batch_size
        #print("MIN_image_bf",torch.min(rgb.float()),"MEAN_image_bf", torch.mean(rgb.float()),"MEDIAN_image_bf", torch.median(rgb.float()),"MAX_image_bf",torch.max(rgb.float()))
        #print("MIN_depth_bf",torch.min(depth.float()).item(),"MEAN_depth_bf", torch.mean(depth.float()).item(),"MEDIAN_depth_bf", torch.median(depth.float()).item(),"MAX_depth_bf",torch.max(depth.float()).item())
        tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
        #output = model(image.to(device),depth.to(device))
        new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                [0.0000, 600.5001220703125, 247.7696533203125],
                [0.0000, 0.0000, 1.0000]])
        new_K = tran(new_K)
        new_K = new_K.to(dtype=torch.float32)

        #print(rgb.shape)
        #print(depth.shape)
        pred = model(rgb.to(device),depth.to(device))
        output_loss = pred
        depth_criterion = criteria.MaskedMSELoss()
        depth_loss = depth_criterion(pred, pcl.to(device))


        loss = depth_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += loss
        average_loss = sum_loss / (epoch_iter / batch_size)
        print('\n', 'Average loss: ', average_loss.item(), ' --- Iterations: ' ,epoch_iter, ' --- Epochs: ', epoch, '\n')
    
    if average_loss < best_loss_and_epoch:

        best_loss_and_epoch = average_loss
        #torch.save(model, 'model_test_best_nn.pth')
        #print('saving model at ',average_loss.item,epoch_iter,epoch)
        print('saving model and weights at ',average_loss.item,epoch_iter,epoch)
        save_batch = (torch.ones(1,3,360,640).to(device), torch.ones(1,1,360,640).to(device))
        #save_batch = {'rgb': torch.ones(1,3,360,640).to(device), 'depth': torch.ones(1,1,360,640).to(device), 'pcl': torch.ones(1,1,360,640).to(device), 'position': torch.zeros(1, 3, 360, 640).to(device), 'K': new_K.to(device)}
        
        with torch.no_grad():
            trace_model = torch.jit.trace(model,save_batch)
        torch.jit.save(trace_model,'decnet_model_and_weights.pth')
    training_duration = training_start_time - time.time()
    print('Epoch training duration: ', training_duration)
print('Total training duration: ', training_duration)

if evaluation == True:
    model = torch.jit.load('decnet_model_and_weights.pth')
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

            rgb = testing((torch.from_numpy(np.array(rgb))),mean_rgb,std_rgb,batch_size,color_type=3)       
            depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d,1,color_type=1)
            pcl = testing((torch.from_numpy(np.array(pcl))),mean_gt,std_gt,1,color_type=1)

            
            rgb = rgb.to(dtype=torch.float32)

            epoch_iter += 1
            #print("MIN_image_bf",torch.min(image.float()),"MEAN_image_bf", torch.mean(image.float()),"MEDIAN_image_bf", torch.median(image.float()),"MAX_image_bf",torch.max(image.float()))
            #print("MIN_depth_bf",torch.min(depth.float()).item(),"MEAN_depth_bf", torch.mean(depth.float()).item(),"MEDIAN_depth_bf", torch.median(depth.float()).item(),"MAX_depth_bf",torch.max(depth.float()).item())

            #print(rgb.shape)
            #print(depth.shape)
            pred = model(rgb.to(device),depth.to(device))
            output_loss = pred
            depth_criterion = criteria.MaskedMSELoss()
            depth_loss = depth_criterion(pred, pcl.to(device))


            loss = depth_loss
            
            sum_loss += loss
            average_loss = sum_loss / (epoch_iter / batch_size)
            progress = 100*epoch_iter/len(mini_test_set)
            print('\n', 'Average loss: ', average_loss.item(), ' --- Progress: ' , progress , '%\n')

        