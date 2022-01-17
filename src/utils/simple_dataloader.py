
'''

import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print(transform)

transform = transforms.Compose([
    # resize
    #transforms.Resize(32),
    # center-crop
    #transforms.CenterCrop(32),
    # to-tensor
    transforms.ToTensor(),
    # normalize
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print(transform)


trainloader = torch.utils.data.DataLoader(trainset,batch_size=8,shuffle=True)

DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 )

'''

from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


class simpleDataloader(Dataset):
    
    def __init__(self, rgb_dir, depth_cm_dir, pcl_cm_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, depth_cm_dir, pcl_cm_dir) for f in rgb_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
    
    def combine_files(self, rgb_file: Path, depth_cm_dir, pcl_cm_dir):
        
        files = {'rgb': rgb_file, 
                 'd': depth_cm_dir/rgb_file.name.replace('rgb', 'depth_cm'),
                 'gt': pcl_cm_dir/rgb_file.name.replace('rgb', 'pcl_cm')}

        return files

    def __len__(self):
        
        return len(self.files)

    
    def open_as_array(self, idx, invert=False, include_depth=False):
        #print(self.files[idx]['rgb'])
        raw_rgb = np.array(Image.open(self.files[idx]['rgb']))
        #raw_rgb = cv2.imread(str(self.files[idx]['rgb']), -1)

        if include_depth:
            depth = np.expand_dims(np.array(Image.open(self.files[idx]['d'])), 2)
            raw_rgb = np.concatenate([raw_rgb, depth], axis=2)

        if invert:
                raw_rgb = raw_rgb.transpose((2,0,1))
        
        
        # normalize
        #print(type(raw_rgb))
        
        #print(raw_rgb.dtype)
        #print("MIN_dataloader",np.min(raw_rgb))
        #print("MEAN_dataloader", np.mean(raw_rgb))
        #print("MEDIAN_dataloader", np.median(raw_rgb))
        #print("MAX_dataloader",np.max(raw_rgb))
        #print(np.iinfo(raw_rgb.dtype).max)
        #print(np.iinfo(raw_rgb.dtype).max)

        #return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    

        return raw_rgb

    def open_mask(self, idx, add_dims=False):
        
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        #raw_mask = (raw_mask / np.iinfo(raw_mask.dtype).max)

        #raw_mask = np.where(raw_mask==255, 1, 0)
        #print(raw_mask.dtype)        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_depth=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=True), dtype=torch.float32)
        #print(x.shape)
        #print(y.shape)
        
        return x, y

    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s



base_path = Path('../../../dataset_nn_24a_original/mini_dataset')

data = simpleDataloader(base_path/'rgb_cropped',
                     base_path/'depth_cm_cropped',
                     base_path/'pcl_cm_cropped')


x, y = data[1000]
print(x)
split_ratio = 0.2
print(len(data))
testing_image = int(len(data)*split_ratio)


train_ds, test_ds = torch.utils.data.random_split(data, ((len(data)-int(len(data)*split_ratio), int(len(data)*split_ratio))))

train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)
'''
'''
fig, ax = plt.subplots(1,2, figsize= (10,9))
ax[0].imshow(data.open_as_array(15))
ax[1].imshow(data.open_mask(15))
'''

'''
print("MIN_yb",torch.min(yb.float()))
print("MEAN_yb", torch.mean(yb.float()))
print("MEDIAN_yb", torch.median(yb.float()))
print("MAX_yb",torch.max(yb.float()))

#plt.imshow(data.open_mask(500))
#plt.show()
image =  '../../../../nn_dataset/pcl_cm/1636964987303228855_pcl_cm.png'
#plt.imshow('../../../../nn_dataset/pcl_cm/1636964987605797291_pcl_cm.png')    
#plt.show()

#print(yb.shape)
detached = yb.numpy()
print(np.squeeze(detached).shape)
print(type(detached))
pcl = cv2.imread(image, -1)
print(pcl.shape)
normalized_pcl =  cv2.normalize(np.squeeze(detached), np.squeeze(detached), 255, 0, cv2.NORM_MINMAX)
#cv2.show()
#raw_mask = np.array(Image.open(image))
cv2.namedWindow("Input")
cv2.imshow("Input", np.array(normalized_pcl, dtype = np.uint8))
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()
#plt.imshow(raw_mask)
#lt.show()


print("MIN_cloud",np.min(pcl))
print("MEAN_cloud", np.mean(pcl))
print("MEDIAN_cloud", np.median(pcl))
print("MAX_cloud",np.max(pcl))
'''