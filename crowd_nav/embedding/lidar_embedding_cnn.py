import torch
import torch.nn as nn
import torchvision.transforms as transforms


class LidarEmbeddingCNN(nn.Module):
    def __init__(self, input_shape):
        super(LidarEmbeddingCNN, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=(1,2), padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        with torch.no_grad():
            self.conv_output_shape = self._get_conv_output_shape(input_shape)

        self.fc = nn.Linear(self.conv_output_shape, 65)

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

    def _get_conv_output_shape(self, shape):
        # Helper function to calculate the output shape of the convolutional layers
        x = torch.rand(*shape).unsqueeze(0)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)

        return self.flatten(x).shape[1]


def run(image_np):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image_np).type(torch.float32)

    model = LidarEmbeddingCNN(image_tensor.size())
    model.eval()

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))

    return output


