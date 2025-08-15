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

csv_root = "finetune_survival/features/clinical_features/shared/"
dl_img_ref_csv = "ssl_pretraining/results/0220_vqvae/codebook/2025-02-19 14:59:59/output_features.csv"
ref_df = pd.read_csv(dl_img_ref_csv)
valid_img_names = ref_df.iloc[:, 0].astype(str).tolist()

def merge_csv_files(external_dataset, input_folder, dl_df, apply_strata=False):
    csv_files = [(file, file.split('_')[0]) for file in os.listdir(input_folder) if file.endswith('.csv') and external_dataset not in file and 'all' not in file]
    data_frames = []

    for (file, center) in csv_files:
        file_path = os.path.join(input_folder, file)

        df = pd.read_csv(file_path)
        df['img_name'] = df['img_name'].astype(str)
        dl_df['img_name'] = dl_df['img_name'].astype(str)

        if apply_strata:
            df['dataset_name'] = center

        df = pd.merge(df, dl_df, on='img_name', how='left')
        df = df.dropna(subset=dl_df.columns[1:])

        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)

    return combined_df

def merge_for_test(test_clinical_df, test_dl_df, dataset_name, apply_strata=False):
    test_clinical_df['img_name'] = test_clinical_df['img_name'].astype(str)
    test_dl_df['img_name'] = test_dl_df['img_name'].astype(str)

    if apply_strata:
        test_clinical_df['dataset_name'] = dataset_name

    df = pd.merge(test_clinical_df, test_dl_df, on='img_name', how='left')
    df = df.dropna(subset=test_dl_df.columns[1:])
    return df

def clean(df, ehr_group, threshold=0.7, apply_strata=False):
    df = clean_and_process_csv(df, columns_to_fill_zero = ['微循环障碍','心肌内出血','心室血栓','西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                            '他汀', '二甲双胍', '列奈'], apply_strata=apply_strata)

    ehr_columns = [
        "心肌内出血", "心室血栓", "室壁瘤",
        "匹林", "他汀", "培哚普利叔丁胺/沙坦", "洛尔", "降糖药", "GENDER", "AGE",
        "Height", "Weight", "BMI", "BSA", "TNIpeak", "甘油三酯", "总胆固醇",
        "高密度脂蛋白", "低密度脂蛋白", "载脂蛋白A", "糖化血红蛋白", "Smoke",
        "Killips", "HBP", "Diabetes", "Dyslipidemia", "SP", "DP",
        "发病时长（单位小时）", "是否累及LAD", "TIMI_grade_pre", "TIMI_grade_post",
        "HR", "EDV", "ESV", "SV", "EF", "Mass"
    ]

    ehr_indexer = {
        'clinical': [0, 1, 2, 22, 23, 24, 25, 29, 21, 28, 30, 31, 32],
        'biochemical': [14, 15, 16, 17, 18, 19, 20],
        'physiological': [9, 8, 10, 11, 12, 26, 27, 33, 34, 35, 36, 37],
        'pharmacological': [3, 4, 5, 6, 7]
    }

    ehr_group = ehr_group.lower()
    groups = ehr_group.split('_')
    if all(g in ehr_indexer for g in groups):
        selected_indices = []
        for g in groups:
            selected_indices.extend(ehr_indexer[g])

        selected_indices = sorted(set(selected_indices), key=selected_indices.index)
        cols = [ehr_columns[i] for i in selected_indices]
    else:
        cols = ehr_columns

    cols_to_keep = list(df.columns[:3]) + list(df.columns[41:]) + [col for col in df.columns[3:] if col in cols]

    df = df[cols_to_keep]

    df = remove_multicollinearity(df, threshold=threshold)

    df = df[df['img_name'].astype(str).isin(valid_img_names)]

    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])
    return df

def clean_for_test(df, apply_strata=False):
    df = clean_and_process_csv(df, columns_to_fill_zero = ['微循环障碍','心肌内出血','心室血栓', '西洛他唑', '波立达', '曲美他嗪', '胰岛素', '华法林',
                                                            '单硝酸异山梨酯', '诺欣妥', '沙班', '倍林达', '噻嗪',
                                                            '特苏敏', '螺内酯', '匹林', '氯吡格雷', '替格瑞洛',
                                                            '洛尔', '波立维', '立普妥', '培哚普利叔丁胺', '地平',
                                                       '他汀', '二甲双胍', '列奈'], apply_strata=apply_strata)

    df = df[df['img_name'].astype(str).isin(valid_img_names)]

    df = df[df['img_name'] != 'wo']
    df = df.dropna(subset=['survival_time'])
    return df

def internal_process(external_dataset, input_folder, dl_df, apply_strata=False):
    csv_files = [(file, file.split('_')[0]) for file in os.listdir(input_folder) if file.endswith('.csv') and external_dataset in file]
    data_frames = []

    for (file, center) in csv_files:
        file_path = os.path.join(input_folder, file)

        df = pd.read_csv(file_path)
        df['img_name'] = df['img_name'].astype(str)
        dl_df['img_name'] = dl_df['img_name'].astype(str)

        if apply_strata:
            df['dataset_name'] = center

        df = pd.merge(df, dl_df, on='img_name', how='left')
        df = df.dropna(subset=dl_df.columns[1:])
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def split(df, seed):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    return train_df, val_df

def split_internal(df, seed, dataset = "wo"):
    trainval_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(trainval_df, test_size=0.2 / (0.6 + 0.2), random_state=seed, shuffle=True)

    train_img_names = train_df["img_name"].tolist()
    val_img_names = val_df["img_name"].tolist()
    test_img_names = test_df["img_name"].tolist()
    data = {
        "train_img_names": train_img_names,
        "val_img_names": val_img_names,
        "test_img_names": test_img_names
    }
    storage_dir = "finetune_survival/features/public_split"
    with open(f"{storage_dir}/img_split_{seed}_{dataset}.json", 'w') as f:
        json.dump(data, f)
    return train_df, val_df, test_df

def proceed(dataset, dl_csv_path, ehr_group, threshold, apply_strata=False, seed=37):
    dl_df = pd.read_csv(dl_csv_path)
    train_df, val_df = split(clean(merge_csv_files(dataset, csv_root, dl_df, apply_strata=apply_strata), ehr_group, threshold, apply_strata=apply_strata), seed)
    test_df = clean_for_test(merge_for_test(pd.read_csv(f"{csv_root}/{dataset}_selected_clinical_features.csv"), dl_df, dataset, apply_strata=apply_strata), apply_strata=apply_strata)

    common_columns = [col for col in train_df.columns if col in val_df.columns and col in test_df.columns]
    train_df = train_df[common_columns]
    val_df = val_df[common_columns]
    test_df = test_df[common_columns]

    return train_df, val_df, test_df, clean(merge_csv_files(dataset, csv_root, dl_df), ehr_group)

def proceed_internal(dataset, dl_csv_path, ehr_group, threshold, apply_strata=False, seed=37):
    dl_df = pd.read_csv(dl_csv_path)
    train_df, val_df, test_df = split_internal(clean(internal_process(dataset, csv_root, dl_df, apply_strata=apply_strata), ehr_group, threshold, apply_strata=apply_strata), seed, dataset)

    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = proceed("gulou", threshold=0.8)
    print(train_df.shape)
    print(test_df.shape)
    print(set(train_df.columns) - set(test_df.columns))
    print(test_df.columns)

