import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import glob
import os

def find_higher_pix(image):
    prev_y = 10000000
    for x in range(image.width):
        for y in range(image.height):
            # for the given pixel at w,h, lets check its value against the threshold
            if image.getpixel((x,y)) > 0: #note that the first parameter is actually a tuple object
                if y < prev_y:
                    prev_y = y
                # lets set this to zero
            else:
                # otherwise lets set this to 255
                pass    #now we just return the new image
    return prev_y
 
    
def find_lower_pix(image):
    prev_y = 0
    for x in range(image.width):
        for y in range(image.height):
            # for the given pixel at w,h, lets check its value against the threshold
            if image.getpixel((x,y)) > 0: #note that the first parameter is actually a tuple object
                if y > prev_y:
                    prev_y = y
                # lets set this to zero
            else:
                # otherwise lets set this to 255
                pass    #now we just return the new image
    return prev_y

folder_loc = '../../../dataset_nn_24a_original'
total_files = glob.glob(os.path.join(folder_loc,'pcl_cm/*'))
#print(total_files)


#os.mkdir('../../../../nn_dataset/dataset_nn24a/mini_dataset/rgb_cropped')
#os.mkdir('../../../../nn_dataset/dataset_nn24a/mini_dataset/depth_cm_cropped')
#os.mkdir('../../../../nn_dataset/dataset_nn24a/mini_dataset/pcl_cm_cropped')
#print(len(total_files))
i = 0
for image_file in sorted(total_files):        
    
    #print(image_file,len(total_files))
    file_name = image_file.lstrip(folder_loc).rstrip('.png').split('/')[-1]
    orig_rgb = Image.open(image_file)
    #orig_depth = Image.open(os.path.join(folder_loc,'pcl_cm_cropped/',file_name+'_pcl_cm_cropped.png')))
    #print(file_name)
    file_name = file_name.rstrip('_pcl_cm')
    #cropped_rgb_file = os.path.join(folder_loc,'rgb_cropped/'+file_name+'_rgb_cropped.png')
    #cropped_depth_file = os.path.join(folder_loc,'depth_cm_cropped/'+file_name+'_depth_cm_cropped.png')
    #cropped_pcl_file = os.path.join(folder_loc,'pcl_cm_cropped/'+file_name+'_pcl_cm_cropped.png')
    #im = Image.open(total_files[0])
    width, height = orig_rgb.size
    #print(width,height)
    #cropped_rgb = Image.open(cropped_rgb_file)
    orig_rgb = Image.open(os.path.join(folder_loc,'rgb/',file_name+'_rgb.png'))
    orig_depth = Image.open(os.path.join(folder_loc,'depth_cm/',file_name+'_depth_cm.png'))
    orig_pcl = Image.open(os.path.join(folder_loc,'pcl_cm/',file_name+'_pcl_cm.png'))
    width, height = orig_rgb.size


    if i==0:
        print("passed")
        crop_top = find_higher_pix(orig_pcl)
        crop_botton = find_lower_pix(orig_pcl)
        print(crop_top,crop_botton)
        left = 0
        top = crop_top
        right = width
        bottom = crop_botton
        while True:
            
            print('top-botton', bottom-top)
            #TOO LAZY TO FIX THIS
            if (bottom-top) - 360 > 1:
                print("a")
                bottom -= 1
                top += 1
            elif (bottom-top) - 360 == 1:
                top += 1
            elif (bottom-top) - 360 == -1:
                bottom += 1
            elif (bottom-top) - 360 < -1:
                bottom += 1
                top -= 1
            else:
                break

    #else:
    #    print("not_passed")
        
    
    # Cropped image of above dimension
    # (It will not change original image)
    #print(top,bottom,bottom-top)
    cropped_rgb = orig_rgb.crop((left, top, right, bottom))
    cropped_depth = orig_depth.crop((left, top, right, bottom))
    cropped_pcl = orig_pcl.crop((left, top, right, bottom))
    #im1.show()
    '''
    print("ORIGINAL RGB DATA {} {} {} {}".format(np.min(orig_rgb),np.mean(orig_rgb),np.median(orig_rgb),np.max(orig_rgb)))
    print("ORIGINAL DEPTH DATA {} {} {} {}".format(np.min(orig_depth),np.mean(orig_depth),np.median(orig_depth),np.max(orig_depth)))
    print("ORIGINAL PCL DATA {} {} {} {}".format(np.min(orig_pcl),np.mean(orig_pcl),np.median(orig_pcl),np.max(orig_pcl)))
    print("Cropped rgb data {} {} {} {}".format(np.min(cropped_rgb),np.mean(cropped_rgb),np.median(cropped_rgb),np.max(cropped_rgb)))
    print("Cropped depth data {} {} {} {}".format(np.min(cropped_depth),np.mean(cropped_depth),np.median(cropped_depth),np.max(cropped_depth)))
    print("Cropped pcl data {} {} {} {}".format(np.min(cropped_pcl),np.mean(cropped_pcl),np.median(cropped_pcl),np.max(cropped_pcl)))
    '''
    new_width,new_height = cropped_rgb.size
    #print(new_width,new_height)
    cropped_rgb = cropped_rgb.save(os.path.join(folder_loc,'mini_dataset/rgb_cropped/',file_name+'_rgb_cropped.png'))
    cropped_depth = cropped_depth.save(os.path.join(folder_loc,'mini_dataset/depth_cm_cropped/',file_name+'_depth_cm_cropped.png'))
    cropped_pcl = cropped_pcl.save(os.path.join(folder_loc,'mini_dataset/pcl_cm_cropped/',file_name+'_pcl_cm_cropped.png'))

    i+=1
    if i%10==0: 
        print('Progress: {:.2f} %'.format((i*100)/len(total_files)))
    #if i%1000==0:
    #    print('mini_dataset_created')
    #    break

