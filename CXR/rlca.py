import numpy as np
import torch 
import wandb

from model import RLCANet
from criterion import criterion
from dataset import get_dataloader
from utils import seed_everything, wandb_settings
from secret import key

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
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--save_name', type=str, default='model.pth', help='name of the file to save the trained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for the optimizer')
    parser.add_argument('--pretrained', type=bool, default=True, help='transfer learning or not')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--loss_func', type=str, default="RLCA", help='loss function')
    parser.add_argument('--model_name', type=str, default="tv_densenet121", help='model name')
    args = parser.parse_args()
    return args



def main():

    seed_everything(42)

    args = parse_args()

    d = vars(args)
    name = args.model_name + "-" + args.loss_func

    # wandb_settings(key, d, "E-DA-CXR-Loss-Comparsion", "DDCVLAB", name)

    # model = create_model(args.num_classes, args.model_name, args.pretrained)
    model = RLCANet(7,3)
    #state_dict = torch.load("/home/dongdong/Medical-Image-Analysis/CXR/0.8975focal.pth")
    #model.load_state_dict(state_dict, strict=True)
    model.to("cuda")

    num_pos = torch.tensor([67,74,24,106,114,33,29]).cuda()
    num_neg = torch.tensor([1308-67,1308-74,1308-24,1308-106,1308-114,1308-33,1308-29]).cuda()

    pre_loss_func = criterion(args.loss_func, num_pos, num_neg)
    pre_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pre_optimizer, factor=0.1, patience=3)

    trainloader, validloader, testloader = get_dataloader(args.base_root, args.csv_name)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(pre_optimizer, max_lr=1e-4, epochs=args.epochs, steps_per_epoch=len(trainloader))

    best_val_auc = 0
    for epoch in range(args.epochs): 
        epoch_loss = 0
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            train_data, train_labels  = train_data.cuda(), train_labels.cuda().to(dtype=torch.float)
            eta, p = model(train_data)
            loss = pre_loss_func(train_labels, eta, p)
            pre_optimizer.zero_grad()
            loss.backward()
            pre_optimizer.step()
            # scheduler.step()

            epoch_loss += loss.item()
                
            # validation  
            if idx % 100 == 0:
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for _, data in enumerate(validloader):
                        test_data, test_labels = data
                        test_data = test_data.cuda()
                        eta, p = model(test_data)
                        preds = []
                        for i in range(7):
                            preds.append(torch.sum(p[:,i] * eta, dim=1)[...,None])
                        y_pred = torch.cat(preds, dim=1)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())
                    
                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc_mean = np.mean(auc_roc_score(test_true, test_pred)) 
                    model.train()

                    if best_val_auc < val_auc_mean:
                        best_val_auc = val_auc_mean
                        torch.save(model.state_dict(), args.save_name)

                    print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))

        # wandb.log({"Val AUC": val_auc_mean, "Epoch": epoch})
        print(epoch_loss)
        
        scheduler.step(epoch_loss)

    state_dict = torch.load(args.save_name)
    model.load_state_dict(state_dict)

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
    print("AUC: ",np.mean(auc_roc_score(valid_true, valid_pred)))
    print("ACC: ",accuracy_multi(torch.from_numpy(valid_true), torch.from_numpy(valid_pred>0.5)) )

    # wandb.log({"Best Val AUC": np.mean(auc_roc_score(valid_true, valid_pred))})
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
    print("AUC: ",np.mean(auc_roc_score(test_true, test_pred)))
    print("ACC: ",accuracy_multi(torch.from_numpy(test_true), torch.from_numpy(test_pred>0.5)) )

    # wandb.log({"Test AUC": np.mean(auc_roc_score(test_true, test_pred))})
    # print("AP: ",ap(torch.from_numpy(test_true), torch.from_numpy(test_pred>0.5)) )

if __name__ == "__main__":
    main()


