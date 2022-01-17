import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataloader import DecnetDataset
from tqdm import tqdm


split_ratio = 0.2
batch_size = 4
base_path = Path('../../../../nn_dataset/')

data = DecnetDataset(base_path/'rgb',
                     base_path/'depth_cm',
                     base_path/'pcl_cm')
train_ds, test_ds = torch.utils.data.random_split(data, ((len(data)-int(len(data)*split_ratio), int(len(data)*split_ratio))))

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum_rgb, channels_sqrd_sum_rgb, channels_sum_d, channels_sqrd_sum_d, channels_sum_gt, channels_sqrd_sum_gt, num_batches = 0, 0, 0, 0, 0, 0, 0 

    for rgbd, gt in tqdm(loader):
        #print(rgbd.dtype)
        #print(gt.dtype)
        channels_sum_rgb += torch.mean(rgbd[:,0:3,:,:], dim=[0, 2, 3])
        channels_sqrd_sum_rgb += torch.mean(rgbd[:,0:3,:,:] ** 2, dim=[0, 2, 3])
        channels_sum_d += torch.mean(rgbd[:,3:,:,:], dim=[0, 2, 3])
        channels_sqrd_sum_d += torch.mean(rgbd[:,3:,:,:] ** 2, dim=[0, 2, 3])
        channels_sum_gt += torch.mean(gt[:,:,:,:], dim=[0, 2, 3])
        channels_sqrd_sum_gt += torch.mean(gt[:,:,:,:] ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean_rgb = channels_sum_rgb / num_batches
    std_rgb = (channels_sqrd_sum_rgb / num_batches - mean_rgb ** 2) ** 0.5

    mean_d = channels_sum_d / num_batches
    std_d = (channels_sqrd_sum_d / num_batches - mean_d ** 2) ** 0.5

    mean_gt = channels_sum_gt / num_batches
    std_gt = (channels_sqrd_sum_gt / num_batches - mean_d ** 2) ** 0.5

    return mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt 


mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt = get_mean_std(train_dl)
print(mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt)
