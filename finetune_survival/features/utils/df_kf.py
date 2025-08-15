import pandas as pd
from lifelines import CoxPHFitter
import json
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from utils.df_preprocess import clean_and_process_csv, remove_multicollinearity

from datetime import datetime
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

def clean_and_split(work_space, dataset, threshold=0.7, n_splits=5):
    csv = f'finetune_survival/features/{dataset}_clinical_features.csv'
    df = pd.read_csv(csv)

    df = clean_and_process_csv(df, columns_to_fill_zero = ['西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                            '他汀', '二甲双胍', '列奈'])
    df = remove_multicollinearity(df, threshold=threshold)
    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=37)

    fold_data = []

    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_val_df, val_df = train_test_split(train_df, test_size=0.2, random_state=37)

        train_img_names = train_df["img_name"].tolist()
        val_img_names = val_df["img_name"].tolist()
        test_img_names = test_df["img_name"].tolist()

        data = {
            "fold": fold + 1,
            "train_img_names": train_img_names,
            "val_img_names": val_img_names,
            "test_img_names": test_img_names
        }
        fold_data.append(data)

    with open(f"finetune_survival/features/public_split/img_split_crossval.json", 'w') as f:
        json.dump(fold_data, f)

    return fold_data, df
