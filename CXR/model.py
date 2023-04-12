import timm
import torch
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

class EnsembleNet(nn.Module):

    def __init__(self):
        super().__init__()

        m1_state_dict = torch.load("/home/dongdong/Medical-Image-Analysis/CXR/0.8975focal.pth")
        m2_state_dict = torch.load("/home/dongdong/Medical-Image-Analysis/CXR/0.8960.pth")

        self.m1 = CXRNet(model_name="tv_densenet121", num_classes=0)
        self.m2 = CXRNet(model_name="coatnet_1_rw_224", num_classes=0)

        self.m1.load_state_dict(m1_state_dict, strict=False)
        self.m2.load_state_dict(m2_state_dict, strict=False)

        # for params in self.m1.parameters():
        #     params.required_grad = False
        # for params in self.m2.parameters():
        #     params.required_grad = False

        self.head = nn.Sequential(
            nn.Linear(1792, 7)   
        )

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.head(x)
        return x       

if __name__ == "__main__":
    x = torch.randn(1,3,224,224)
    model = EnsembleNet()
    model(x)