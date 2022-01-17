import MinkowskiEngine as ME
import torch
import numpy as np
import time
import torch.nn as nn
from matplotlib import image


print("data_spec")
print("check minkowski speedup")

input_size = (480,640)
input_width, input_height = input_size
#print(input_width, input_height)

depth_img = image.imread('data/nn_dataset/depth_cm/1636964987303228855_depth_cm.png')
depth_img= np.expand_dims(depth_img, axis=0)
depth_img = np.expand_dims(depth_img, axis=0)

#depth_img = np.load('data/nn_dataset/pcl_cm/1636964987303228855_pcl_cm.png',allow_pickle=True)

data = torch.rand(1,1,input_height,input_width,dtype=torch.float64)
data[0][0][0][0] = 0.0
#print(data)


class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                #has_bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)

def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)

#result = to_sparse_coo(data)
#print(result)
#result_2 = to_sparse_coo(data2)

#x = torch.cat(result,result_2,dim=1)
#print(x)

def sparsify_depth(x):
    """
    Sparsify depth map
    Parameters
    ----------
    x : Dense depth map [B,1,H,W]
    Returns
    -------
    Sparse depth map (range values only in valid pixels)
    """
    b, c, h, w = x.shape

    u = torch.arange(w, device=x.device).reshape(1, w).repeat([h, 1])
    v = torch.arange(h, device=x.device).reshape(h, 1).repeat([1, w])
    uv = torch.stack([v, u], 2)

    idxs = [(d > 0)[0] for d in x]

    coords = [uv[idx] for idx in idxs]
    feats = [feats.permute(1, 2, 0)[idx] for idx, feats in zip(idxs, x)]
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    return ME.SparseTensor(coordinates=coords, features=feats, device=x.device)


def densify_features(x, shape):
    """
    Densify features from a sparse tensor
    Parameters
    ----------
    x : Sparse tensor
    shape : Dense shape [B,C,H,W]
    Returns
    -------
    Dense tensor containing sparse information
    """
    stride = x.tensor_stride
    coords, feats = x.C.long(), x.F
    shape = (shape[0], shape[2] // stride[0], shape[3] // stride[1], feats.shape[1])
    dense = torch.zeros(shape, device=x.device, dtype=torch.float64)
    #print(coords,feats)
    #print(dense)
    dense[coords[:, 0],
          coords[:, 1] // stride[0],
          coords[:, 2] // stride[1]] = feats
    return dense.permute(0, 3, 1, 2).contiguous()


start = time.time()
result = sparsify_depth(torch.from_numpy(np.asarray(depth_img)))
shape = np.asarray(data).shape 
print(shape)

print(type(result))
print('got '+ str(result.shape[0]) + ' 3d points out of ' + str((shape[2]*shape[3])) + ' points (dimension of image)')
print(result)

#print(f"Runtime of the program is {time.time() - start}")

#print()

#start = time.time()
#result2 = to_sparse_coo(np.asarray(data2))
#print(f"Runtime of the program is {time.time() - start}")
#print(result2)
#out = densify_features(result, shape)
#print(out.shape)

#ni, n1, n2, n3, n4, n5 = 32, 32, 64, 128, 256, 512
#channels = [ni, n1, n2, n3, n4, n5]

#kernel_sizes = [5, 5] + [3] * (len(channels) - 1)
#print(kernel_sizes)

net = ExampleNetwork(in_feat=1, out_feat=5, D=2)
#print(net)

output = net(result)

#print(output)