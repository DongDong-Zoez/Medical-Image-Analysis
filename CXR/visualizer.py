from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import matplotlib.pyplot as plt

from model import CXRNet
from augmentations import get_transform
import torch

model = CXRNet()
state_dict = torch.load("/home/dongdong/Medical-Image-Analysis/CXR/0.8975focal.pth")
model.load_state_dict(state_dict)
model_children = list(model.features.children())

target_layers = model_children[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

image = plt.imread("/home/dongdong/Medical-Image-Analysis/CXR/xray_jpg/small_pulmonary_nodules/10_0.jpg")

transform = get_transform(train=False, img_size=224, rotate_degree=0)
image = transform(image=image)["image"]
origin = image.numpy().transpose(1,2,0)
origin = origin - origin.min()
origin = origin / origin.max()
image = image.unsqueeze(0)
image = image.to(device)

input_tensor = image
cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

targets = None

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(origin, grayscale_cam, use_rgb=True)
plt.imsave(str('res/feature_maps.jpg'), visualization)