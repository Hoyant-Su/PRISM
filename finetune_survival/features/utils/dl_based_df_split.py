import pandas as pd
from lifelines import CoxPHFitter
import sys
import logging
import io
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from utils.df_preprocess import clean_and_process_csv, remove_multicollinearity
from sklearn.preprocessing import StandardScaler

def clean_and_split(work_space, dataset, dl_csv_path, threshold=0.7):
    csv = f'finetune_survival/features/{dataset}_clinical_features.csv'

    df = pd.read_csv(csv)
    dl_csv = dl_csv_path
    dl_df = pd.read_csv(dl_csv)

    label_columns = dl_df.iloc[:, :1]
    features = dl_df.iloc[:, 1:]

    df['img_name'] = df['img_name'].astype(str)
    df = pd.merge(df, dl_df, on='img_name', how='left')
    df = df.dropna(subset=dl_df.columns[1:])
    df = clean_and_process_csv(df, columns_to_fill_zero = ['心肌内出血','心室血栓','室壁瘤', '西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                            '他汀', '二甲双胍', '列奈'])

    df = remove_multicollinearity(df, threshold=threshold)

    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])

    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=37, shuffle=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2/(0.65 + 0.15), random_state=37, shuffle=True)

    train_img_names = train_df["img_name"].tolist()
    val_img_names = val_df["img_name"].tolist()
    test_img_names = test_df["img_name"].tolist()

    data = {
        "train_img_names": train_img_names,
        "val_img_names": val_img_names,
        "test_img_names": test_img_names
    }

    with open(f"finetune_survival/features/public_split/img_split_renji_7_3.json", 'w') as f:
        json.dump(data, f)

    return train_df, val_df, test_df, df