import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vggnet11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
}

# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel
# RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be
# loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

class VggNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(VggNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vggnet(pretrained=False, model_root=None, **kwargs):
    model = VggNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vggnet11'], model_root))
    return model

if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 224, 224)
    model = VggNet()
    torch.onnx.export(model, dummy_input, "vgg11.onnx")
    print(model)
