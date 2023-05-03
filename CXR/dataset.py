import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from augmentations import get_transform
from utils import split

class CXRDataset(Dataset):

    def __init__(self, df, transforms, normalize=True, mode="train"):
        
        self.img_files = df.path.to_list()
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.normalize = normalize
    
    def __getitem__(self, x):
        row = self.df.iloc[x, :].values
        *_, aortic, arterial, abnorm, lung, spinal, cardiac, inter = row
        image = plt.imread(self.img_files[x])
        # Convert the pixel value to [0, 1]
        if self.normalize:
            image = image - image.min()
            image = image / image.max()
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        if self.mode == "train":
            one_hot_label = [aortic, arterial, abnorm, lung, spinal, cardiac, inter]
            return image, torch.tensor(one_hot_label)
        else:
            return image
    
    def __len__(self):
        return len(self.img_files)
    
def get_dataloader(base_root, csv_name):

    train_transform = get_transform(train=True, img_size=224, rotate_degree=10)
    val_transform = get_transform(train=False, img_size=224, rotate_degree=10)

    df_train, df_valid, df_test = split(base_root, csv_name)

    traindSet = CXRDataset(df_train, train_transform, False)
    validSet = CXRDataset(df_valid, val_transform, False)
    testSet =  CXRDataset(df_test, val_transform, False)
    
    trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=8, shuffle=True)
    validloader =  torch.utils.data.DataLoader(validSet, batch_size=8, shuffle=True)
    testloader =  torch.utils.data.DataLoader(testSet, batch_size=8, shuffle=False)

    return trainloader, validloader, testloader