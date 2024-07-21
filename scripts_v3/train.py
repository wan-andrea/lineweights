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




parent = "C:\\Users\\andre\\Documents\\lineweights\\processed_data\\"

transform = transforms.Compose([
    transforms.Resize((333, 225)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(parent, transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# print(dataset.classes)

train_size = int(0.8 * len(dataset)) # 80%
test_size = len(dataset) - train_size #20%

# test train split
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
# The model, adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB images have 3 channels
        # we want it to recognize up to 6 different patterns in the input
        # 5 pixel window 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(69984, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# define class weights
class_weights = torch.tensor([3.05, 62.5, 5.49, 2.11]) # Calculate frequency of each class, then 1/x to get inverse freq.


criterion = nn.CrossEntropyLoss() # put this between parentheses to use class_weights: weight = class_weights
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Clip gradients?
        nn_utils.clip_grad_norm_(net.parameters(), max_norm=1)

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss / (i+1):.3f}, Training Accuracy: {accuracy:.3f}')


print('Finished Training')
PATH = './large2.pth'
torch.save(net.state_dict(), PATH)

# Test the model
net.eval()  # Set the model to evaluation mode
test_loss = 0
correct = 0

# Load
data_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load the test dataset
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the model to evaluation mode
net.eval()

# Initialize test loss and accuracy
test_loss = 0
correct = 0

# Test the model
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

# Print test loss and accuracy
print(f'Test loss: {test_loss / len(test_data_loader)}')
print(f'Test accuracy: {correct / len(test_dataset):.2f}')




# Utility
"""# Load image
img = Image.open("C:\\Users\\andre\\Documents\\lineweights\\small_processed_data\\Profile\\normal_0_C8_7bc51bd0-97f1-45f4-bb19-47e32e186204.jpg")
#Transform image
transformed_img = transform(img)

# Print shape of image tensor
print("Tensor size:", transformed_img.size()) # [3, 333, 225] channels, height, width

# Visualize the transformed image
transformed_img = transformed_img.permute(1, 2, 0)
plt.imshow(transformed_img)
plt.show()"""