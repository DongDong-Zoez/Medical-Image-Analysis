import numpy as np
import torch 

from model import create_model
from criterion import criterion
from dataset import get_dataloader

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
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--save_name', type=str, default='model.pth', help='name of the file to save the trained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for the optimizer')
    parser.add_argument('--pretrained', type=bool, default=True, help='transfer learning or not')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    args = parser.parse_args()
    return args



def main():

    args = parse_args()

    model = create_model(args.num_classes, args.pretrained)
    model.to("cuda")

    pre_loss_func = criterion()
    pre_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pre_optimizer, factor=0.1, patience=3)

    trainloader, validloader, testloader = get_dataloader(args.base_root, args.csv_name)

    best_val_auc = 0
    for epoch in range(args.epochs): 
        epoch_loss = 0
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            train_data, train_labels  = train_data.cuda(), train_labels.cuda().to(dtype=torch.float)
            y_pred = model(train_data)
            y_pred = torch.sigmoid(y_pred)
            loss = pre_loss_func(y_pred, train_labels)
            pre_optimizer.zero_grad()
            loss.backward()
            pre_optimizer.step()

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
                        y_pred = model(test_data)
                        y_pred = torch.sigmoid(y_pred)
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
    print("AP: ",ap(torch.from_numpy(valid_true), torch.from_numpy(valid_pred>0.5)) )

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
    print("AP: ",ap(torch.from_numpy(test_true), torch.from_numpy(test_pred>0.5)) )

if __name__ == "__main__":
    main()


