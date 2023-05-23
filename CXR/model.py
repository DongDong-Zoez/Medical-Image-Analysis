import timm
import torch
import torch.nn as nn
from TorchCRF import CRF

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
    
class RLCANet(nn.Module):
    def __init__(self, num_labels, num_latent_variable, flexible=False):
        super(RLCANet, self).__init__()
        self.num_labels = num_labels
        self.num_latent_variable = num_latent_variable
        self.flexible = flexible
        self.backbone = timm.create_model("tv_densenet121", num_classes=num_labels, pretrained=False, features_only=True, out_indices=[4])
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        in_channel = self.__get_in_channel()
        self.latent_classifier = nn.Linear(in_channel, num_latent_variable)
        if flexible:
            self.condition_classifier = nn.Linear(in_channel, num_labels * num_latent_variable, bias=False)
        else:
            self.condition_classifier = nn.Linear(in_channel, num_labels, bias=False)
        # The index (m, j) of the matrix represent γ_{mj}
        self.intercept = nn.Parameter(torch.zeros(num_labels, num_latent_variable))

    def __get_in_channel(self):
        x = torch.randn(1,3,224,224)
        return self.backbone(x)[0].shape[1]

        
    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.pool(x).squeeze(-1).squeeze(-1)
        # Here, we make the covariate z as same as the feature vector x
        z = x
        # η_j
        latent = self.latent_classifier(x)
        # p_{mj}
        probs = []
        prob = self.condition_classifier(z)
        if self.flexible:
            pass
        else:
            for j in range(self.num_latent_variable):
                pmj = self.intercept[:,j]+prob
                probs.append(pmj.unsqueeze(-1))
        probs = torch.cat(probs, dim=-1)
        return latent, probs

if __name__ == "__main__":
    x = torch.randn(4,3,224,224)
    y = torch.tensor([[1,0,1,1,0,0,0]])
    model = RLCANet(7,3)
    print(model)
    latent, prob = model(x)
    print(latent.shape, prob.shape)