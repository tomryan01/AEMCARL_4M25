import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np 

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * (self.input_shape[0] // 4) * 112, 960)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        
        return x.squeeze(1)


def run (image_np):  
    # loaded_arr = np.loadtxt("/home/iman/demofile2.txt")
    # image_np = loaded_arr.reshape(
    #     loaded_arr.shape[0], loaded_arr.shape[1] // 4, 4)

    print(np.shape(image_np))

    model = CNN(np.shape(image_np))
    image_np = np.array(image_np).astype(np.float32)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image_np).type(torch.float32)

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0)) 

    print(output)