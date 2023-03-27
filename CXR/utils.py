import random
import numpy as np
import torch
import pandas as pd
import os

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def seed_everything(seed):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set PyTorch deterministic operations for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocessor(base_root, csv_name):
    df = pd.read_csv(os.path.join(base_root, csv_name), index_col=0)
    paths = []
    for _, row in df.iterrows():
        paths.append(os.path.join(base_root, row.category_eng, row.Filename))
    df["path"] = paths
    category = ['主動脈硬化(鈣化)','動脈彎曲','肺野異常','肺紋增加','脊椎病變','心臟肥大','肺尖肋膜增厚']
    df.label = df.label.apply(lambda x: ",".join([cat for cat in category if cat in x]))
    for pathology in category :
        df[pathology] = df.label.apply(lambda x: 1 if pathology in x else 0)
    return df


def split(base_root, csv_name):

    df = preprocessor(base_root, csv_name)
    labels = df.iloc[:, 6:]
    
    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    iterator = iter(kf.split(range(df.shape[0]), labels))
    train_idx, val_idx = next(iterator)

    labels = df.iloc[val_idx,6:]
    kf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    iterator = iter(kf.split(val_idx, labels))
    _val_idx, _test_idx = next(iterator)
    val_idx, test_idx = val_idx[_val_idx], val_idx[_test_idx]

    df_train = df.loc[train_idx]
    df_valid = df.loc[val_idx]
    df_test = df.loc[test_idx]

    return df_train, df_valid, df_test
