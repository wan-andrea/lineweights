import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
# from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


parent = "C:\\Users\\andre\\Documents\\lineweights\\processed_data\\"
# Load the dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = torchvision.datasets.ImageFolder(parent, transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

train_size = int(0.8 * len(dataset)) # 80%
test_size = len(dataset) - train_size #20%

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
model = torchvision.models.resnet50(pretrained=True)
num_classes = 4  # replace with your number of classes
model.fc = torch.nn.Linear(2048, num_classes)

# Define a loss function and an optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the model using your dataset
for epoch in range(10):  # loop over the dataset multiple times
    print("looping1!")
    for i, data in enumerate(data_loader, 0):
        print("looping2!")
        # Get the inputs and labels
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))


