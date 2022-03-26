import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(54*54*16,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,2)

  def forward(self,X):
    X = self.pool(F.relu(self.conv1(X)))
    X = self.pool(F.relu(self.conv2(X)))
    X = X.view(-1, 54*54*16)
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)

class_names = ['CAT', 'DOG']

im = Image.open(r'./static//10000.jpg')
my_transforms = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )])

# image = Image.open(io.BytesIO(image_bytes))
image = my_transforms(im).unsqueeze(0)
image = image.view(1,3,224,224)
# print(image)
model = classifier()
model.load_state_dict(torch.load('model_state.pth', map_location='cpu'))
model.eval()
with torch.no_grad():
    new_pred = model(image).argmax()
print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')