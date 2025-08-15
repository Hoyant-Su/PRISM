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

import sys
sys.path.append("finetune_survival/features/utils")

from df_preprocess import clean_and_process_csv, remove_multicollinearity

csv_root = "finetune_survival/features/clinical_features/shared"
dl_img_ref_csv = "ssl_pretraining/results/0220_vqvae/codebook/2025-02-19 14:59:59/output_features.csv"
ref_df = pd.read_csv(dl_img_ref_csv)
valid_img_names = ref_df.iloc[:, 0].astype(str).tolist()

def merge_csv_files(external_dataset, input_folder, apply_strata=False):
    csv_files = [(file, file.split('_')[0]) for file in os.listdir(input_folder) if file.endswith('.csv') and external_dataset not in file]
    data_frames = []

    for (file, center) in csv_files:
        file_path = os.path.join(input_folder, file)

        df = pd.read_csv(file_path)
        if apply_strata:
            df['dataset_name'] = center

        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def clean(df, threshold=0.7, apply_strata=False):
    df = clean_and_process_csv(df, columns_to_fill_zero = ['微循环障碍','心肌内出血','心室血栓', '西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                            '他汀', '二甲双胍', '列奈'], apply_strata=apply_strata)

    df = remove_multicollinearity(df, threshold=threshold)

    df = df[df['img_name'].astype(str).isin(valid_img_names)]

    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])
    return df

def clean_for_test(df, dataset, apply_strata=False):
    if apply_strata:
        df['dataset_name'] = dataset

    df = clean_and_process_csv(df, columns_to_fill_zero = ['微循环障碍','心肌内出血','心室血栓','西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                            '他汀', '二甲双胍', '列奈'], apply_strata=apply_strata)

    df = df[df['img_name'].astype(str).isin(valid_img_names)]
    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])
    return df

def split(df, seed):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    return train_df, val_df

def split_internal(df, seed):
    trainval_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(trainval_df, test_size=0.2 / (0.6 + 0.2), random_state=seed, shuffle=True)
    return train_df, val_df, test_df

def proceed(dataset, threshold, apply_strata=False, seed=37):
    train_df, val_df = split(clean(merge_csv_files(dataset, csv_root, apply_strata=apply_strata), threshold, apply_strata=apply_strata), seed)
    test_df = clean_for_test(pd.read_csv(f"{csv_root}/{dataset}_selected_clinical_features.csv"), dataset, apply_strata=apply_strata)

    common_columns = [col for col in train_df.columns if col in val_df.columns]
    train_df = train_df[common_columns]
    val_df = val_df[common_columns]
    test_df = test_df[common_columns]

    return train_df, val_df, test_df, clean(merge_csv_files(dataset, csv_root))

def proceed_internal(dataset, threshold, apply_strata=False, seed=37):
    train_df, val_df, test_df = split_internal(clean(pd.read_csv(f"{csv_root}/{dataset}_selected_clinical_features.csv"), threshold=threshold, apply_strata=apply_strata), seed)

    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = proceed("gulou", threshold=0.8)
    print(train_df.shape)
    print(test_df.shape)
    print(set(train_df.columns) - set(test_df.columns))
    print(test_df.columns)
