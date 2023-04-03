import numpy as np
import torch 
import wandb

from model import create_model
from criterion import criterion
from dataset import get_dataloader
from utils import seed_everything

from libauc.metrics import auc_roc_score
from fastai.metrics import accuracy_multi, RocAucMulti, APScoreMulti

roc_auc = RocAucMulti(average=None)
mean_auc = RocAucMulti(average="macro")
ap = APScoreMulti()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    parser.add_argument('--base_root', type=str, default='/home/dongdong/Medical-Image-Analysis/CXR/xray_jpg/', help='path to the base data directory')
    parser.add_argument('--csv_name', type=str, default='chest_dongdong.csv', help='name of the CSV file containing the dataset')
    parser.add_argument('--pretrained_path', type=str, default='0.8603.pth', help='name of the file to save the trained model')
    parser.add_argument('--pretrained', type=bool, default=True, help='transfer learning or not')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    args = parser.parse_args()
    return args



def main():

    seed_everything(42)

    args = parse_args()

    model = create_model(args.num_classes, args.pretrained)
    model.to("cuda")

    trainloader, validloader, testloader = get_dataloader(args.base_root, args.csv_name)

    state_dict = torch.load(args.pretrained_path)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        train_pred = []
        train_true = []
        for _, data in enumerate(trainloader):
            train_data, train_labels = data
            train_data = train_data.cuda()
            y_pred = model(train_data)
            y_pred = torch.sigmoid(y_pred)
            train_pred.append(y_pred.cpu().detach().numpy())
            train_true.append(train_labels.numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        val_auc_mean = np.mean(auc_roc_score(train_true, train_pred))

    # show auc roc scores for each task 
    print("*" * 10, " Training ", "*" * 10)
    print("AUC: ",auc_roc_score(train_true, train_pred))
    print("AUC: ",np.mean(auc_roc_score(train_true, train_pred)))
    print("ACC: ",accuracy_multi(torch.from_numpy(train_true), torch.from_numpy(train_pred>0.5)) )

    model.eval()
    with torch.no_grad():
        valid_pred = []
        valid_true = []
        for _, data in enumerate(validloader):
            valid_data, valid_labels = data
            valid_data = valid_data.cuda()
            y_pred = model(valid_data)
            y_pred = torch.sigmoid(y_pred)
            valid_pred.append(y_pred.cpu().detach().numpy())
            valid_true.append(valid_labels.numpy())

        valid_true = np.concatenate(valid_true)
        valid_pred = np.concatenate(valid_pred)
        val_auc_mean = np.mean(auc_roc_score(valid_true, valid_pred))

    # show auc roc scores for each task 
    print("*" * 10, " Validation ", "*" * 10)
    print("AUC: ",auc_roc_score(valid_true, valid_pred))
    print("AUC: ",np.mean(auc_roc_score(valid_true, valid_pred)))
    print("ACC: ",accuracy_multi(torch.from_numpy(valid_true), torch.from_numpy(valid_pred>0.5)) )
    # print("AP: ",ap(torch.from_numpy(valid_true), torch.from_numpy(valid_pred>0.5)) )

    with torch.no_grad():
        test_pred = []
        test_true = []
        for _, data in enumerate(testloader):
            test_data, test_labels = data
            test_data = test_data.cuda()
            y_pred = model(test_data)
            y_pred = torch.sigmoid(y_pred)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_labels.numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        val_auc_mean = np.mean(auc_roc_score(test_true, test_pred))

    # show auc roc scores for each task 
    print("*" * 10, " Test ", "*" * 10)
    print("AUC: ",auc_roc_score(test_true, test_pred))
    print("AUC: ",np.mean(auc_roc_score(test_true, test_pred)))
    print("ACC: ",accuracy_multi(torch.from_numpy(test_true), torch.from_numpy(test_pred>0.5)) )
    # print("AP: ",ap(torch.from_numpy(test_true), torch.from_numpy(test_pred>0.5)) )

if __name__ == "__main__":
    main()


