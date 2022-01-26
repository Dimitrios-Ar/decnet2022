import glob
import numpy as np
import os
from PIL import Image
#import mean_std_custom
from pathlib import Path
#from dataloader import DecnetDataset
from torch.utils.data import Dataset, DataLoader, sampler
from imutils import paths
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms

#from mean_std_custom import get_mean_std as mean_std
'''
def SanityDatasetCheckss(folder_loc):

    print("[INFO] loading image paths...")
    rgbPath = list(paths.list_images(os.path.join(folder_loc,'rgb_cropped')))
    depthPath = list(paths.list_images(os.path.join(folder_loc,'depth_cm_cropped')))
    pclPath = list(paths.list_images(os.path.join(folder_loc,'pcl_cm_cropped')))
    print(len(rgbPath),len(depthPath),len(pclPath))
'''
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((352,608)),
    transforms.PILToTensor()
    ])


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum_rgb, channels_sqrd_sum_rgb, channels_sum_d, channels_sqrd_sum_d, channels_sum_gt, channels_sqrd_sum_gt, num_batches = 0, 0, 0, 0, 0, 0, 0 

    for rgb,depth,pcl in tqdm(loader):
        #print(rgbd.dtype)
        #print(gt.dtype)
        rgb_float = rgb.type(torch.FloatTensor)
        #print(rgb_float.shape)

        rgb_float = torch.transpose(rgb_float, 3,1)
        #print(rgb_float.shape)
        depth_float = depth.type(torch.FloatTensor)
        depth_float = depth_float[:,:,:,None]
        depth_float = torch.transpose(depth_float, 3,1)
        #print(depth_float.shape)
        #depth_float = torch.transpose(depth_float, 0,2)
        #print(depth_float.shape)
        #depth_float = depth_float[:,:,:,None]
        pcl_float = pcl.type(torch.FloatTensor)
        pcl_float = pcl_float[:,:,:,None]
        pcl_float = torch.transpose(pcl_float, 3,1)
#print(depth_float.shape)
        #print(depth.shape)
        channels_sum_rgb += torch.mean(rgb_float[:,:,:,:], dim=[0, 2, 3])
        channels_sqrd_sum_rgb += torch.mean(rgb_float[:,:,:,:] ** 2, dim=[0, 2, 3])
        channels_sum_d += torch.mean(depth_float[:,:,:,:], dim=[0, 2, 3])
        channels_sqrd_sum_d += torch.mean(depth_float[:,:,:,:] ** 2, dim=[0, 2, 3])
        channels_sum_gt += torch.mean(pcl_float[:,:,:,:], dim=[0, 2, 3])
        channels_sqrd_sum_gt += torch.mean(pcl_float[:,:,:,:] ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean_rgb = channels_sum_rgb / num_batches
    std_rgb = (channels_sqrd_sum_rgb / num_batches - mean_rgb ** 2) ** 0.5

    mean_d = channels_sum_d / num_batches
    std_d = (channels_sqrd_sum_d / num_batches - mean_d ** 2) ** 0.5

    mean_gt = channels_sum_gt / num_batches
    std_gt = (channels_sqrd_sum_gt / num_batches - mean_d ** 2) ** 0.5

    return mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt 

def visualize_batch(batch):
	# initialize a figure
	fig = plt.figure("{} batch".format("dataset_type"),
    figsize=(50, 50),dpi=100)
	# loop over the batch size
	for i in range(0, 4):
		# create a subplot
		ax = plt.subplot(4, 4, i+1)
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		plt.imshow(image)
		plt.axis("off")
		bx = plt.subplot(4, 4, i + 5)
		depth = batch[1][i].cpu().numpy()
		plt.imshow(depth)
		plt.axis("off")
		pcl = batch[2][i].cpu().numpy()
		bx = plt.subplot(4, 4, i + 9)

        #depth = 
		#image = image.transpose((1, 2, 0))
		#image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		#idx = batch[1][i]
		#label = classes[idx]
		# show the image along with the label
		plt.imshow(pcl)
		#plt.title(label)
		plt.axis("off")
	# show the plot
	#plt.tight_layout()
	plt.show()

class SanityDatasetCheck(Dataset):
    def __init__(self, rgbPathLoc,depthPathLoc,pclPathLoc):
        #'Initialization'
        self.rgbPathLoc = rgbPathLoc
        self.depthPathLoc = depthPathLoc
        self.pclPathLoc = pclPathLoc
    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.rgbPathLoc)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        rgb_file = self.rgbPathLoc[index]
        depth_file = self.depthPathLoc[index]
        pcl_file = self.pclPathLoc[index]
        #depth_file = 
        #files = {'rgb': rgb_file, 
        #         'depth': depth_file,
        #         'pcl': pcl_file}
        
        #ID = self.files[index]['rgb']
        #print(ID)
        # Load data and get label
        raw_rgb = np.array(Image.open(rgb_file))
        #print(raw_rgb.shape)
        depth = np.array(Image.open(depth_file))
        pcl = np.array(Image.open(pcl_file))
        image_id = rgb_file

        #print(raw_rgb)
        #depth = np.expand_dims(np.array(Image.open(self.files[idx]['d'])), 2)

        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]
        raw_rgb = transform(raw_rgb)
        depth = transform(depth)
        pcl = transform(pcl)

        return raw_rgb, depth, pcl, image_id



'''
class SanityDatasetCheck(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def total_image_loader(self,dir):
            images = []
            dir = os.path.expanduser(dir)
            for target in sorted(os.listdir(dir)):
                d = os.path.join(dir, target)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if is_image_file(fname):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)
            return images
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #rgb,depth,pcl = self.

        
        
        img_name = os.path.join(self.root_dir,
                                 '')
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample
'''
def data_sanity_image_size(folder_loc):
    total_files = glob.glob(folder_loc)
    #print(len(total_files))

    i = 0
    width_array = np.empty(1)
    height_array = np.empty(1)
    for image_file in sorted(total_files):
        file_name = image_file.lstrip(folder_loc).rstrip('.png').split('/')[-1]
        orig_rgb = Image.open(image_file)
        file_name = file_name.rstrip('_pcl_cm_cropped') 
        width, height = orig_rgb.size
        width_array = np.append(width_array, width)
        height_array = np.append(height_array, height)

    width_array = np.delete(width_array, 0)
    height_array = np.delete(height_array, 0)
    print('main_folder and amount of dataset', folder_loc, len(total_files), )
    print('mean and std of width', np.mean(width_array),np.std(width_array))
    print('mean and std of height', np.mean(height_array),np.std(height_array))


#def testbatch_loader(): 
    #testbatch_folder = '../../../Desktop/data_sanity/testbatch'

#data_sanity_image_size('../../../Desktop/data_sanity/nn_dataset_24a_cropped/pcl_cm_cropped/*')
#dataloader_simple('../../../Desktop/data_sanity/nn_dataset_24a_cropped/rgb_cropped',
#                    '../../../Desktop/data_sanity/nn_dataset_24a_cropped/depth_cm_cropped',
##                    '../../../Desktop/data_sanity/nn_dataset_24a_cropped/pcl_cm_cropped')
#print("[INFO] loading image paths...")
'''
folder_loc = '../../../Desktop/data_sanity/testbatch'
rgbPath = list(paths.list_images(os.path.join(folder_loc,'rgb_cropped')))
depthPath = list(paths.list_images(os.path.join(folder_loc,'depth_cm_cropped')))
pclPath = list(paths.list_images(os.path.join(folder_loc,'pcl_cm_cropped')))
sanity_set = SanityDatasetCheck(rgbPath,depthPath,pclPath)
train_dl = DataLoader(sanity_set, batch_size=10)

print(len(train_dl))
nextbatch = next(iter(train_dl))
mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt = get_mean_std(train_dl)
print(mean_rgb, std_rgb, mean_d, std_d, mean_gt, std_gt)
visualize_batch(nextbatch)
'''