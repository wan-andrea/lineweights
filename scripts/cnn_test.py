# The Grasshopper script will take Q curves,
    # (1) open all closed curves
    # (2) rebuild them with a fixed number of N control points and M degree, resulting in P knot vectors

# Then store the following:
    # (1) Each control point is a 3D-tuple (x, y, z), resulting in a Q length list of N length list of 3D-tuples
    # (2) A Q length list of N-length list of integers representing the weight of each control point
    # (3) A P length list of N-length list of floats representing the knot vector


# So If I fix the curve control points at 10, and degree at 3
# control points: [[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)....(x10, y10, z10)]] # means there will be 30 elements per curve
# weights: [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)...] # means there will be 10 elements per curve
# knot vector [(1, 2, 3, 4, 5, 6, 7, 8, 9 10, 11, 12)] # means there will be 12 elements per curve

# Goal: Predict control points from given a normal map image

# Init
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Helper functions 
# function to open files and store as data
# Inputs: file location as str
# Outputs: data as a 2D list

def pklToLst(fileLocation):
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data

# First try - dataset containing normal map images, and control points
# Definitions
class normalMapDataset(Dataset):
    # ctrl_pts (3D list with 3D tuples as elements): contains the pkl file with the control points
    # (example: [[[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)....(x10, y10, z10)], [(x1, y1, z1)...(x10, y10, z10)]]])
    # ctrl_pts[i] == a drawing
    # ctrl_pts[i][j] == a curve within a drawing
    # ctrl_pts[i][j][k] = a control point on a curve
    # norm_dir (string): directory with all the normal map images

    def __init__(self, ctrl_data, norm_dir, transform=None):
        self.ctrl_pts = pklToLst(ctrl_data)
        self.norm_dir = norm_dir
    
    # size of dataset == number of normal map drawings
    def __len__(self):
        return len(os.listdir(self.norm_dir))
    
    # such that dataset[i] gets the ith sample
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        
        # Get the file name of the normal map image
        img_name = os.listdir(self.norm_dir)[i]
    
        # Read the normal map image
        img_path = os.path.join(self.norm_dir, img_name)
        image = io.imread(img_path)
        
        # Get the corresponding control points
        ctrl_pts = self.ctrl_pts[i]
        # Creates a 4D numpy, where:
            # ctrl_pts[i] is the ith drawing
            # ctrl_pts[i][j] is the jth curve in the ith drawing
            # ctrl_pts[i][j][k] is the kth control point (as a list [x, y, z]) in the jth curve of the ith drawing
        ctrl_pts = np.array(ctrl_pts)
        
        # Create a dictionary with the image and control points
        sample = {'image': image, 'ctrl_pts': ctrl_pts}
        
        return sample

# Create dataset
# the file location of the pickled file, the directory to the normal map images
normal_dataset = normalMapDataset(ctrl_pkl = "curves.pkl", norm_dir = "normal_maps\\")

# Preprocess images
class ToTensor(object):
    # Converts multi-dimensional arrays in sample to Tensors. 
    # (Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    def __call__(self, sample):
        image, ctrl_pts = sample['image'], sample['ctrl_pts']
        # swap colour axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2 0 1))
        return {'image': torch.from_numpy(image), 'ctrl_pts': torch.from_numpy(ctrl_pts)}

# Convert dataset to tensor
# Inputs: ctrl_pkl file path (string), norm_dir file path (string)
# Outputs: the dataset, with multi-dimensional arrays converted to Tensors
def datasetToTensor(ctrl_pkl, norm_dir):
    return normalMapDataset(ctrl_pkl = ctrl_pkl, norm_dir = norm_dir, transform=transforms.Compose([ToTensor()]))

transformed_dataset = datasetToTensor("curves.pkl", norm_dir = "normal_maps\\")


# Load the dataset
def loadDataset(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

loadDataset(transformed_dataset, 1, True)

# Creating the model

# Get the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 500 * 3)  # Output dimension: 500 curves per drawing, 3 coordinates per curve

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
