
# Imports
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
# from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torchcam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class ImageRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        folders = ['Cut', 'Profile', 'Contour', 'Detail']
        labels = [4, 3, 2, 1]
        for folder, label in zip(folders, labels):
            folder_path = os.path.join(self.image_folder, folder)
            images = os.listdir(folder_path)
            self.images.extend([os.path.join(folder_path, image) for image in images])
            self.labels.extend([label] * len(images))

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)

transform = transforms.Compose([
        transforms.Resize((300), transforms.InterpolationMode.LANCZOS, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset = ImageRegressionDataset(image_folder='C:\\Users\\andre\\Documents\\lineweights\\processed_data', transform=transform)

# Define class weights by calculating inverse frequency of each class
class_weights = torch.tensor([3.05, 62.5, 5.49, 2.11])
criterion = nn.MSELoss()

# Splitting dataset into train and test datasets
print("Splitting dataset into training and testing datasets.")
train_percent = 0.8
train_size = int(train_percent * len(dataset)) # 80%
test_size = len(dataset) - train_size #20%
print(f"The training dataset is {train_percent*100}% of the dataset.")

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB images have 3 channels
        # we want it to recognize up to 6 different patterns in the input
        # 5 pixel window 
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=9, padding=4)
        self.conv5 = nn.Conv2d(24, 30, kernel_size=11, padding=5)
        self.conv6 = nn.Conv2d(30, 36, kernel_size=13, padding=6)
        self.conv7 = nn.Conv2d(36, 42, kernel_size=15, padding=7)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(69984, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = x.view(-1, 5544000)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

net.train()

for epoch in range(10):
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        mae = torch.sum(torch.abs(outputs - labels)) / len(labels)
        mse = torch.sum((outputs - labels) ** 2) / len(labels)
        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}, MAE: {mae.item():.4f}, MSE: {mse.item():.4f}')

net.eval()
with torch.no_grad():
    for i, data in enumerate(test_data_loader):
        inputs, labels = data
        outputs = net(inputs)
        mae = torch.sum(torch.abs(outputs - labels)) / len(labels)
        mse = torch.sum((outputs - labels) ** 2) / len(labels)
        print(f'Test Batch {i+1}, MAE: {mae.item():.4f}, MSE: {mse.item():.4f}')
