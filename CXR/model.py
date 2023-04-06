import timm
import torch.nn as nn

class CXRNet(nn.Module):

    def __init__(self, num_classes=7, model_name="tv_densenet121", pretrained=True):
        super().__init__()

        self.features = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.features(x)


def create_model(num_classes, model_name, pretrained_path):
    model = CXRNet(num_classes, model_name, pretrained_path)
    return model

if __name__ == "__main__":
    model  = CXRNet(7)
    print(model)