import io
import torch
import torch.nn as nn 
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


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

def get_model():
    checkpoint = './model_state.pth'
    model = classifier()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict = False)
    model.eval()
    return model

def get_tensor(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )])
    image = Image.open(io.BytesIO(image_bytes))
    image = my_transforms(image).unsqueeze(0)
    image = image.view(1,3,224,224)
    return image

def get_output(image_bytes):
     with torch.no_grad():
        model = get_model()
        tensor = get_tensor(image_bytes)
        prediction = model(tensor).argmax()
        #prediction = model(image).argmax()
        prediction = class_names[prediction.item()]
     return prediction

# timg = Image.open(r'./static//10000.jpg')
# data = get_output(image_bytes=timg)
# print(data)