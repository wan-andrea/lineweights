import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import os
import sys
from PIL import Image
import pickle

# Specify the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pklToLst(fileLocation):
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data

# ENCODER USING RESNET50
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        resnet.to(device)
        for param in resnet.parameters():
            param.requires_grad_(False)
    
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn1 = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn1(features)
        features.to(device)
        
        return features

# DECODER
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, output_size):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output):
        hidden_state = self.init_hidden(encoder_output.size(0))
        output, _ = self.lstm(encoder_output, hidden_state)
        output = self.linear(output[:, -1, :])  # Use the last time step's output
        return output

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to() # device
        return hidden_state


# Create a dataset class for images and curves
class ImageCurveDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, curve_folder):
        self.image_folder = image_folder
        self.curve_folder = curve_folder

    def __getitem__(self, index):
        image_file = os.listdir(self.image_folder)[index]
        curve_file = os.listdir(self.curve_folder)[index]

        image = Image.open(os.path.join(self.image_folder, image_file))
        curve = pklToLst(os.path.join(self.curve_folder, curve_file))
        curve = torch.tensor(curve)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = transform(image)

        return image, curve

    def __len__(self):
        return len(os.listdir(self.image_folder))

def train(num_epochs, encoder, decoder, device, image_folder, curve_folder):
    # Create a dataset and sampler
    dataset = ImageCurveDataset(image_folder, curve_folder)
    sampler = torch.utils.data.SubsetRandomSampler(range(len(dataset)))

    # Create a data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, sampler=sampler)


    # Train the model
    for epoch in range(1, num_epochs+1):
        for images, curves in data_loader:
            images = images.to(device)
            curves = curves.to(device)
            features = encoder(images)
            features = features.to(device)
            generated_curves = decoder(features)

            loss = criterion(generated_curves, curves)
            loss.backward()
            optimizer.step()

            # Print the loss at each epoch
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Create instances of the encoder and decoder
encoder = EncoderCNN(embed_size=256)
encoder = encoder.to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, num_layers=1, output_size=52)
decoder=decoder.to(device)

# preprocess data, image to 256 x 256
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.ToTensor()]),                           # convert the PIL Image to a tensor)

# Specify the image folder, curve folder
image_folder = 'img_data\\'
curve_folder = 'crv_data\\' # x1, y1, z1, x2, y2, z2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

# Train the model
train(num_epochs=10, encoder=encoder, decoder=decoder, device=device, image_folder=image_folder, curve_folder=curve_folder)

print("All done!")