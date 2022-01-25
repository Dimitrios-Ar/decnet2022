import os
from pathlib import Path
from imutils import paths
import glob

folder_loc = Path('../../../Desktop/data_sanity/nn_dataset_24k_original/nn_dataset_24k_cropped')

rgbPath = list(paths.list_images(os.path.join(folder_loc,'rgb_cropped')))
depthPath = list(paths.list_images(os.path.join(folder_loc,'depth_cm_cropped')))
pclPath = list(paths.list_images(os.path.join(folder_loc,'pcl_cm_cropped')))
print(rgbPath)
total_files = glob.glob(os.path.join(folder_loc,'rgb_cropped/*'))
print(len(sorted(total_files)))

moving_files = sorted(total_files)[420:]
print(len(moving_files))