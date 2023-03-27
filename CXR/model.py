import timm
import torch.nn as nn

class CXRNet(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.features = timm.create_model("tv_densenet121", num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.features(x)


def create_model(num_classes, pretrained_path):
    model = CXRNet(num_classes, pretrained_path)
    return model