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
from features.utils.df_preprocess import clean_and_process_csv, remove_multicollinearity

dl_img_ref_csv = "ssl_pretraining/results/0220_vqvae/codebook/2025-02-19 14:59:59/output_features.csv"
ref_df = pd.read_csv(dl_img_ref_csv)
valid_img_names = ref_df.iloc[:, 0].astype(str).tolist()

def clean_and_split(work_space, dataset, seed=37, threshold=0.7):
    csv = f'finetune_survival/features/clinical_features/shared/{dataset}_selected_clinical_features.csv'

    df = pd.read_csv(csv)
    df = clean_and_process_csv(df, columns_to_fill_zero = ['西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                            '他汀', '二甲双胍', '列奈'])

    df = remove_multicollinearity(df, threshold=threshold)

    df = df[df['img_name'].astype(str).isin(valid_img_names)]

    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])

    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2/(0.65 + 0.15), random_state=seed, shuffle=True)

    train_img_names = train_df["img_name"].tolist()
    val_img_names = val_df["img_name"].tolist()
    test_img_names = test_df["img_name"].tolist()

    data = {
        "train_img_names": train_img_names,
        "val_img_names": val_img_names,
        "test_img_names": test_img_names
    }

    with open(f"finetune_survival/features/public_split/img_split_gulou.json", 'w') as f:
        json.dump(data, f)

    return train_df, val_df, test_df, df
