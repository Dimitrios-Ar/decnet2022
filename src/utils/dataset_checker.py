import glob
import numpy as np
import os
from PIL import Image

folder_loc = '../../../dataset_nn_24a_original/mini_dataset'
total_files = glob.glob(os.path.join(folder_loc,'pcl_cm_cropped/*'))
print(len(total_files))

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
print('main_folder and amount of dataset', os.path.join(folder_loc,'pcl_cm_cropped/'), len(total_files), )
print('mean and std of width', np.mean(width_array),np.std(width_array))
print('mean and std of height', np.mean(height_array),np.std(height_array))

#print(height_array)
