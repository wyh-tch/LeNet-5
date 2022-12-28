import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

modelName = 'googlenet.onnx'

# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel
# RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be
# loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7,
                               stride=2, padding=3)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1_1, red_3_3, out_3_3, red_5_5, out_5_5, out_1_1pool
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.inception3b(self.inception3a(x)))
        x = self.maxpool4(self.inception4e(self.inception4d(self.inception4c(self.inception4b(self.inception4a(x))))))
        x = self.inception5b(self.inception5a(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



class Inception(nn.Module):
    def __init__(self, in_channels, out_1_1, red_3_3, out_3_3, red_5_5, out_5_5, out_1_1pool):
        super(Inception, self).__init__()
        self.branch1 = conv_block(in_channels, out_1_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3_3, kernel_size=1),
            conv_block(red_3_3, out_3_3, kernel_size=3, padding=1)
            )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5_5, kernel_size=1),
            conv_block(red_5_5, out_5_5, kernel_size=5, padding=2)
            )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1_1pool, kernel_size=1)
            )

    def forward(self, x):
        # concatenate the outputs of the four branches
        # N x (out_1×1 + out_3×3 + out_5×5 + out_1×1pool) x 28 x 28
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



def googlenet(pretrained=False, model_root=None, **kwargs):
    model = GoogLeNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet'], model_root))
    return model



if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = GoogLeNet()
    torch.onnx.export(model, dummy_input, modelName)
    print(model)