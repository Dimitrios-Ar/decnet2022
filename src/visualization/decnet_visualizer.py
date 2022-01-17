import os
import matplotlib.pyplot as plt
import open3d
import cv2
import numpy as np
import pyquaternion as pyq
from scipy.spatial.transform import Rotation as R
import torch
from torchvision import transforms
import ros_numpy
import glob
from PIL import Image
from datetime import datetime
import rospy
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as rosImage
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
 

# =========================================================
# TEST DIMARA
# =========================================================
def build_camera_info(seq_i):  # pylint: disable=no-self-use
    
    camera_info = CameraInfo()
    # store info without header
    camera_info.header = Header()
    camera_info.header.stamp = rospy.Time.now()
    camera_info.header.frame_id = "l515"
    camera_info.header.seq = seq_i
    #camera_info.header = None
    camera_info.width = 640
    camera_info.height = 480
    camera_info.distortion_model = 'plumb_bob'
    cx = 318.6040344238281
    cy = 247.7696533203125
    fx = 599.9778442382812
    fy = 600.5001220703125
    camera_info.K = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
    camera_info.D = [0., 0., 0., 0., 0.]
    camera_info.R = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    camera_info.P = [fx, 0, cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]

    return camera_info 

def return_pred_msg(data):
    #ros_mgs = rosImage()
    ros_mgs = ros_numpy.msgify(rosImage, data, encoding='16UC1')

    ros_mgs.header = Header()
    ros_mgs.header.stamp = rospy.Time.now()
    ros_mgs.header.frame_id = "l515"

    ros_mgs.header.seq = seq_i
    #ros_mgs.width = 640
    #ros_mgs.height = 480
    return ros_mgs
    
def return_rgb_msg(data):
    #ros_mgs = rosImage()
    orig_img_msg = ros_numpy.msgify(rosImage, data, encoding='rgb8')
            
    orig_img_msg.header = Header()
    orig_img_msg.header.stamp = rospy.Time.now()
    orig_img_msg.header.frame_id = "l515"

    #camera_info.header.frame_id = "map"
    orig_img_msg.header.seq = seq_i
    #ros_mgs.width = 640
    #ros_mgs.height = 480
    return orig_img_msg

def testing(img,mean,std):
   img = img.type(torch.FloatTensor)
   transform_norm = transforms.Normalize((mean),(std))
   #print(img.dtype)
   i=0
   normalized_batch = torch.zeros(1,1,480,640)
   for element in img:
      
       #print(element)
       img_normalized = transform_norm(element)
       normalized_batch[i] = img_normalized
       i+=1
   return normalized_batch

def denormalize(img,mean,std):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [0],
                                                     std = [1/std]),
                                transforms.Normalize(mean = [ -mean ],
                                                     std = [ 1. ]),
                               ])

    inv_tensor = invTrans(img)
    return inv_tensor


def minmaxmedian(velo):
    all_values =(velo.min(axis=0),np.median(velo,axis=0),velo.max(axis=0))
    return all_values

def lidar_to_cam_transform(cam_tra,cam_rot):

    r = R.from_euler('zyx', cam_rot, degrees=False)
    trans = np.zeros((4,4))
    trans[0:3, 0:3] = r.as_matrix()
    trans[0:3, 3] = cam_tra
    trans[3, 3] = 1
    #print(trans)

    return trans

def project_velodyne_to_l515(calib):
    velodyne_to_l515_transform = lidar_to_cam_transform(cam_tra=[-0.0504, -0.019, 0.0655], cam_rot = [1.5409, -0.01, 1.5950])
    R_ref2rect = np.eye(4)
    P_rect2cam2 = np.array([599.9778442382812, 0.0, 318.6040344238281, 0.0, 0.0, 600.5001220703125, 247.7696533203125, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4)
    proj_mat = P_rect2cam2 @ R_ref2rect @ velodyne_to_l515_transform
    #print(P_rect2cam2)
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]

# =========================================================
# Utils
# =========================================================

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    #print('original scan', scan)
    scan = scan.reshape((-1, 4))
    #print('later_scan', scan[0])
    return scan

def load_velo_scan_pcd(velo_filename):
    #velodyne_pcd = open3d.io.read_point_cloud("velo_filename")
    scan = np.load(velo_filename)
    #print(scan.shape)
    #print(scan[0],scan[1],scan[2],scan[3],scan[4],scan[5])
    #scan = scan.reshape((-1, 3))
    
    #print(scan)
    #scan = np.asarray(velodyne_pcd.points)
    #scan = scan.reshape((-1, 4))
    #print(scan)
    #scan = scan.reshape((-1, 4))
    #print(scan)
    #print(scan)

    return scan


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def project_camera_to_lidar(points, proj_mat):
    """
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]

    Returns:
        points in lidar coordinate:     [3, npoints]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    return points[:3, :]

def render_lidar_on_image(pts_velo, img, calib, img_width, img_height, depth_img):
    print("MIN_cv2_img_depth",np.min(depth_img))
    print("MEAN_cv2_img_depth", np.mean(depth_img))
    print("MEDIAN_cv2_img_depth", np.median(depth_img))
    print("MAX_cv2_img_depth",np.max(depth_img))

    proj_velo2cam2 = project_velodyne_to_l515(calib)

    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    print(img_width,img_height)
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap_rgb = plt.cm.get_cmap('hsv', 256) #which colormap to get and steps (values)
    cmap_rgb = np.array([cmap_rgb(i) for i in range(256)])[:, :3] * 255

    cmap_depth = plt.cm.get_cmap('gray', 256) #which colormap to get and steps (values)
    cmap_depth = np.array([cmap_depth(i) for i in range(256)])[:, :3] * 255
    

    blank_image = np.zeros((img_height,img_width,1), np.float32)

   

    for i in range(imgfov_pc_pixel.shape[1]):

        
        #print(i)
        depth = imgfov_pc_cam2[2, i]
        #print('depth_value',depth)
        #print((int(np.round(imgfov_pc_pixel[0, i])),int(np.round(imgfov_pc_pixel[1, i]))))
        try:
            #blank_image[int(np.round(imgfov_pc_pixel[0, i])), int(np.round(imgfov_pc_pixel[1, i]))] = depth*1000
        #print(int(np.round(imgfov_pc_pixel[0, i])), int(np.round(imgfov_pc_pixel[1, i])), depth)
        #break
            rgb_color = cmap_rgb[int(100.0 / depth), :]

            depth_color = cmap_depth[int(100.0 / depth), :]#print(color)
            #print('depth_color', depth_color)

            cv2.circle(blank_image, (int(np.round(imgfov_pc_pixel[0, i])),
                           int(np.round(imgfov_pc_pixel[1, i]))),
                    2, color=int(depth*100), thickness=-1)
                    

            cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                            int(np.round(imgfov_pc_pixel[1, i]))),
                    2, color=tuple(rgb_color), thickness=-1)
        except:
            pass
    print('blank_image_values', blank_image.min(),np.median(blank_image),blank_image.max())    
    '''
    fig = plt.figure(figsize=(10, 7))
    
    rows = 2
    columns = 1

    fig.add_subplot(rows, columns, 1)

    plt.imshow(depth_img)
    plt.axis('off')
    plt.title("depth")

    fig.add_subplot(rows, columns, 2)

    plt.imshow(img)
    plt.axis('off')
    plt.title("rgb")


    
    # resize image
    
    #plt.imshow(resized)
    plt.yticks([])
    plt.xticks([])
    #plt.draw()
    #plt.pause(0.1)
    '''

    #print(blank_image)
    return img,blank_image

if __name__ == '__main__':
    
    rospy.init_node('visualizer')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    folder_loc = '../../../../nn_dataset/dataset_nn24a'
    total_files = glob.glob(os.path.join(folder_loc,'rgb/*'))
    print(total_files)
    model_test = torch.load('../weights/last_saved_model_in_the_zeropoint5range.pth')

    mean_rgb = torch.tensor([103.1967,  99.8189,  80.7972])
    std_rgb = torch.tensor([51.7836, 47.0790, 40.0790])
    mean_d = torch.tensor(116.8378)
    std_d = torch.tensor(162.2317)
    mean_gt = torch.tensor(72.2703)
    std_gt = torch.tensor(258.6499)

    seq_i = 0
    pred_pub = rospy.Publisher('depth_registered/image_rect', rosImage, queue_size=5)
    camera_info_pub = rospy.Publisher('rgb/camera_info', CameraInfo, queue_size=5)
    orig_pub = rospy.Publisher('rgb/image_rect_color', rosImage, queue_size=5)
    #print(total_files)
    test_single = False

    if test_single == True:
        now = datetime.now()

        
        image_file = '../../../../nn_dataset/dataset_nn24a/rgb/1638881607081797123_rgb.png' #PALLET


        #image_file = '../../../../nn_dataset/dataset_nn24a/rgb/1638881467991832256_rgb.png'# manual forklift
        #image_file = '../../../../nn_dataset/dataset_nn24a/rgb/1638881062044625759_rgb.png'# test
        #image_file = '../../../../nn_dataset/dataset_nn24a/rgb/1638881100470455170_rgb.png'# full
        #image_file = '../../../../nn_dataset/dataset_nn24a/rgb/1638881102688850880_rgb.png'# full

        file_name = image_file.lstrip(folder_loc).rstrip('.png').split('/')[-1]
        print(file_name)
        file_name = file_name.rstrip('_rgb')
        test_rgb = os.path.join(folder_loc,'rgb/'+file_name+'_rgb.png')
        test_depth = os.path.join(folder_loc,'depth_cm/'+file_name+'_depth_cm.png')
        test_pcl = os.path.join(folder_loc,'pcl_cm/'+file_name+'_pcl_cm.png')

        rgb = cv2.cvtColor(cv2.imread(os.path.join(folder_loc,'rgb/'+file_name+'_rgb.png')), cv2.COLOR_BGR2RGB)#Need to change png
        depth_img = cv2.imread(os.path.join(folder_loc,'depth_cm/'+file_name+'_depth_cm.png'), -1)

        img_height,img_width, img_channel = rgb.shape

        pc_velo = load_velo_scan_pcd(os.path.join(folder_loc,'pcl/'+file_name+'_pcl.npy'))#[:, :3]

        calib = ''
        returned_image, pcl_to_image = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height,depth_img)
        
        scale_percent = 100 # percent of original size
        width = int(returned_image.shape[1] * scale_percent / 100)
        height = int(returned_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(returned_image, dim, interpolation = cv2.INTER_AREA)
        
        #cv2.imwrite('dataset_rgbd_test_again/pcl_cm/'+file_name.split('_')[0]+'_pcl_cm.png', pcl_to_image.astype(np.uint16))

        #cv2.normalize(pcl_to_image, pcl_to_image, 0, 255, cv2.NORM_MINMAX)
        
        # Save your OpenCV2 image as a jpeg 
        #cv2.imwrite('dataset_rgbd_test_again/pcl_visual/'+file_name.split('_')[0]+'_pcl_visual.png', pcl_to_image.astype(np.uint8))
        #np.save('dataset_rgbd_test_again/pcl_mm/'+file_name+'_pcl_mm.npy', pcl_to_image)



        test_depth = np.array(Image.open(test_depth))                                                    
        raw_rgb = np.array(Image.open(test_rgb))
        depth = np.expand_dims(test_depth, 2)
        gt = np.expand_dims(np.array(Image.open(test_pcl)), 2)
        #print(raw_rgb.shape,depth.shape)

        #raw_rgb = np.concatenate([raw_rgb, depth], axis=2)
        #raw_rgb = (raw_rgb / np.iinfo(raw_rgb.dtype).max)

        #print(raw_rgb.shape)
        #raw_rgb = np.expand_dims(raw_rgb,axis=0)
        #raw_rgb = raw_rgb.transpose((2,0,1))
        #raw_rgb = np.expand_dims(raw_rgb,axis=0)

        image, depth, gt  = np.expand_dims(raw_rgb.transpose(2,0,1),axis=0), np.expand_dims(depth.transpose(2,0,1),axis=0), np.expand_dims(gt.transpose(2,0,1),axis=0)

        depth_tensor = torch.from_numpy(test_depth)
        depth_tensor = depth_tensor.type(torch.FloatTensor)
        #np.array(Image.open(test_depth))
        print("depth_bef",torch.min(depth_tensor), torch.mean(depth_tensor), torch.median(depth_tensor),torch.max(depth_tensor))


        #print(image.shape,depth.shape,gt.shape)
        depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d)
        gt = testing((torch.from_numpy(np.array(gt))),mean_gt,std_gt)
        image_tensor = torch.from_numpy(np.asarray(image))
        image_tensor = image_tensor.type(torch.FloatTensor)

        #print("depth_norm",torch.min(depth), torch.mean(depth), torch.median(depth),torch.max(depth))

        denormalized = denormalize(depth,mean_d,std_d)
        #print("depth_denormalized",torch.min(denormalized), torch.mean(denormalized), torch.median(denormalized),torch.max(denormalized))


        tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
        new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                [0.0000, 600.5001220703125, 247.7696533203125],
                [0.0000, 0.0000, 1.0000]])
        new_K = tran(new_K)
        new_K = new_K.to(dtype=torch.float32)
        batch_data = {'rgb': image_tensor.to(device), 'd': depth.to(device), 'g': gt.to(device), 'position': torch.zeros(1, 3, 480, 640).to(device), 'K': new_K.to(device)} 
        st1_pred, st2_pred, pred = model_test(batch_data)
        output = pred

        denormalized_output = denormalize(output, mean_d, std_d)
        duration = datetime.now() - now
        print(duration)
        final_depth = denormalized_output.detach().to('cpu').numpy()
        final_depth = final_depth.squeeze()
        final_depth[final_depth > 3000] = 0
        print(test_depth.shape)
        predicted_image = (255*(final_depth - np.min(final_depth))/np.ptp(final_depth)).astype(np.uint8)     
        depth_image = (255*(test_depth - np.min(test_depth))/np.ptp(test_depth)).astype(np.uint8)
        
        predicted_image_color = cv2.applyColorMap(predicted_image, cv2.COLORMAP_HSV)

        cv2.imshow('Depth extrapolation network prediction', predicted_image_color)
        cv2.imshow("Close-range depth", depth_image)
        cv2.imshow("Projected lidar on image", cv2.cvtColor(resized,cv2.COLOR_RGB2BGR))


        key = cv2.waitKey(0)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows(0)
            
        
            

    else: 
        for image_file in sorted(total_files):
            now = datetime.now()

            #current_time = now.strftime("%H:%M:%S")
            #print("Current Time =", current_time)
            print(image_file)
            file_name = image_file.lstrip(folder_loc).rstrip('.png').split('/')[-1]
            print(file_name)
            file_name = file_name.rstrip('_rgb')
            test_rgb = os.path.join(folder_loc,'rgb/'+file_name+'_rgb.png')
            test_depth = os.path.join(folder_loc,'depth_cm/'+file_name+'_depth_cm.png')
            test_pcl = os.path.join(folder_loc,'pcl_cm/'+file_name+'_pcl_cm.png')

            rgb = cv2.cvtColor(cv2.imread(os.path.join(folder_loc,'rgb/'+file_name+'_rgb.png')), cv2.COLOR_BGR2RGB)#Need to change png
            depth_img = cv2.imread(os.path.join(folder_loc,'depth_cm/'+file_name+'_depth_cm.png'), -1)

            img_height,img_width, img_channel = rgb.shape

            pc_velo = load_velo_scan_pcd(os.path.join(folder_loc,'pcl/'+file_name+'_pcl.npy'))#[:, :3]

            calib = ''
            returned_image, pcl_to_image = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height,depth_img)
            
            scale_percent = 100 # percent of original size
            width = int(returned_image.shape[1] * scale_percent / 100)
            height = int(returned_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(returned_image, dim, interpolation = cv2.INTER_AREA)
            
            #cv2.imwrite('dataset_rgbd_test_again/pcl_cm/'+file_name.split('_')[0]+'_pcl_cm.png', pcl_to_image.astype(np.uint16))

            #cv2.normalize(pcl_to_image, pcl_to_image, 0, 255, cv2.NORM_MINMAX)
            
            # Save your OpenCV2 image as a jpeg 
            #cv2.imwrite('dataset_rgbd_test_again/pcl_visual/'+file_name.split('_')[0]+'_pcl_visual.png', pcl_to_image.astype(np.uint8))
            #np.save('dataset_rgbd_test_again/pcl_mm/'+file_name+'_pcl_mm.npy', pcl_to_image)



            test_depth = np.array(Image.open(test_depth))                                                    
            raw_rgb = np.array(Image.open(test_rgb))
            depth = np.expand_dims(test_depth, 2)
            gt = np.expand_dims(np.array(Image.open(test_pcl)), 2)
            #print(raw_rgb.shape,depth.shape)

            #raw_rgb = np.concatenate([raw_rgb, depth], axis=2)
            #raw_rgb = (raw_rgb / np.iinfo(raw_rgb.dtype).max)

            #print(raw_rgb.shape)
            #raw_rgb = np.expand_dims(raw_rgb,axis=0)
            #raw_rgb = raw_rgb.transpose((2,0,1))
            #raw_rgb = np.expand_dims(raw_rgb,axis=0)

            image, depth, gt  = np.expand_dims(raw_rgb.transpose(2,0,1),axis=0), np.expand_dims(depth.transpose(2,0,1),axis=0), np.expand_dims(gt.transpose(2,0,1),axis=0)

            depth_tensor = torch.from_numpy(test_depth)
            depth_tensor = depth_tensor.type(torch.FloatTensor)
            #np.array(Image.open(test_depth))
            print("depth_bef",torch.min(depth_tensor), torch.mean(depth_tensor), torch.median(depth_tensor),torch.max(depth_tensor))


            #print(image.shape,depth.shape,gt.shape)
            depth = testing((torch.from_numpy(np.array(depth))),mean_d,std_d)
            gt = testing((torch.from_numpy(np.array(gt))),mean_gt,std_gt)
            image_tensor = torch.from_numpy(np.asarray(image))
            image_tensor = image_tensor.type(torch.FloatTensor)

            #print("depth_norm",torch.min(depth), torch.mean(depth), torch.median(depth),torch.max(depth))

            denormalized = denormalize(depth,mean_d,std_d)
            #print("depth_denormalized",torch.min(denormalized), torch.mean(denormalized), torch.median(denormalized),torch.max(denormalized))


            tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
            new_K = np.array([[599.9778442382812, 0.0000, 318.6040344238281],
                    [0.0000, 600.5001220703125, 247.7696533203125],
                    [0.0000, 0.0000, 1.0000]])
            new_K = tran(new_K)
            new_K = new_K.to(dtype=torch.float32)
            batch_data = {'rgb': image_tensor.to(device), 'd': depth.to(device), 'g': gt.to(device), 'position': torch.zeros(1, 3, 480, 640).to(device), 'K': new_K.to(device)} 
            st1_pred, st2_pred, pred = model_test(batch_data)
            output = pred

            denormalized_output = denormalize(output, mean_d, std_d)
            duration = datetime.now() - now
            print(duration)
            final_depth = denormalized_output.detach().to('cpu').numpy()
            final_depth = final_depth.squeeze()
            final_depth[final_depth > 3000] = 0
            print(test_depth.shape)
            predicted_image = (255*(final_depth - np.min(final_depth))/np.ptp(final_depth)).astype(np.uint8)     
            depth_image = (255*(test_depth - np.min(test_depth))/np.ptp(test_depth)).astype(np.uint8)
            
            predicted_image_color = cv2.applyColorMap(predicted_image, cv2.COLORMAP_HSV)

            cv2.imshow('Depth extrapolation network prediction', predicted_image_color)
            cv2.imshow("Close-range depth", depth_image)
            cv2.imshow("Projected lidar on image", cv2.cvtColor(resized,cv2.COLOR_RGB2BGR))
            #cv2.imshow("Projected lidar on image", cv2.cvtColor(resized,cv2.COLOR_RGB2BGR))
            
            cv2.imwrite('example/nn_dataset_lidar.png', cv2.cvtColor(resized[40:420, :],cv2.COLOR_RGB2BGR))
            cv2.imwrite('example/nn_dataset_depth.png', depth_image[40:420, :])
            cv2.imwrite('example/nn_dataset_prediction.png', predicted_image_color[40:420, :])
            cv2.imwrite('example/nn_dataset_rgb.png', cv2.cvtColor(raw_rgb[40:420, :],cv2.COLOR_RGB2BGR))
            

            key = cv2.waitKey(1)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
            seq_i += 1
            camera_info = build_camera_info(seq_i)
            #print(camera_info)
            print(type(predicted_image))
            print(final_depth.shape)
            pred_img_msg = return_pred_msg(final_depth.astype(np.uint16))
            #pred_img_msg = ros_numpy.msgify(rosImage, final_depth.astype(np.uint16), encoding='mono16')
            orig_img_msg = return_rgb_msg(raw_rgb)
            #orig_img_msg = ros_numpy.msgify(rosImage, raw_rgb, encoding='rgb8')
            
            pred_pub.publish(pred_img_msg)
            camera_info_pub.publish(camera_info) 
            orig_pub.publish(orig_img_msg)

            
        print("done")