import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np 

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()

        with torch.no_grad():
            self.conv_output_shape = self._get_conv_output_shape(input_shape)

        self.fc = nn.Linear(self.conv_output_shape, 65)
    
    def _get_conv_output_shape(self, shape):
        #Helper function to calculate the output shape of the convolutional layers
        x = torch.rand(*shape)
        print(x.size())
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        return self.flatten(x).shape[1]
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        
        return x.squeeze(1)


#passing the image through to test whether it works
def run (image_np):  
    # loaded_arr = np.loadtxt("/home/iman/demofile2.txt")
    # image_np = loaded_arr.reshape(
    #     loaded_arr.shape[0], loaded_arr.shape[1] // 4, 4)
    # print("image shape:",image_np.shape)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(image_np).type(torch.float32).unsqueeze(0)

    # print(type(image_tensor.size()))
    # print(image_tensor.size())
    inp_size = torch.Size([1, 4, 10, 450])
    # model = CNN(image_tensor.size())
    model = CNN(inp_size)
    image_np = np.array(image_np).astype(np.float32)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor) 

    return output

def get_embedding(lidar_image, model):

    # print("lidar image shape:",lidar_image.shape)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(lidar_image).type(torch.float32).unsqueeze(0)
    
    if model is None:
        inp_size = torch.Size([1, 4, 10, 450])
        # model = CNN(image_tensor.size())
        model = CNN(inp_size)

    lidar_image = np.array(lidar_image).astype(np.float32)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor) 

    return model, output


