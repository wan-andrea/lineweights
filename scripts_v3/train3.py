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

# Folder to dataset
parent = "C:\\Users\\andre\\Documents\\lineweights\\processed_data\\"

transform = transforms.Compose([
    # Resize image with variable aspect ratio using bilinear, bicubic, or lanczos
    transforms.Resize((300), transforms.InterpolationMode.LANCZOS, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(parent, transform)

# Contour, Cut, Detail, Profile
print(f"The dataset classes are: {dataset.classes}")

# Define class weights by calculating inverse frequency of each class
class_weights = torch.tensor([3.05, 62.5, 5.49, 2.11])

# Splitting dataset into train and test datasets
print("Splitting dataset into training and testing datasets.")
train_percent = 0.8
train_size = int(train_percent * len(dataset)) # 80%
test_size = len(dataset) - train_size #20%
print(f"The training dataset is {train_percent*100}% of the dataset.")

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)

# Defining the model
print("Now, the model will be defined.")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=9, padding=4)
        self.conv5 = nn.Conv2d(24, 30, kernel_size=11, padding=5)
        self.conv6 = nn.Conv2d(30, 36, kernel_size=13, padding=6)
        self.conv7 = nn.Conv2d(36, 42, kernel_size=15, padding=7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 42)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Usage
net = Net()

print("Define the optimizer and loss function.")
criterion = nn.CrossEntropyLoss(class_weights)
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("Training begins.")
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_data_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Print loss and accuracy at each batch
        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}')
        
        # Accumulate loss
        running_loss += loss.item()
    
# Print average loss and accuracy over an epoch
print(f'Epoch {epoch+1}, Average Loss: {running_loss/i:.4f}, Average Accuracy: {correct/total:.4f}')

print('Finished Training')


# Evaluate the model on the test set
print("Now the model will be evaluated.")
net.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data in test_data_loader:
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(test_data_loader)
print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss / len(test_data_loader), accuracy))
