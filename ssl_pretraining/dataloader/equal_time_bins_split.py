import pandas as pd
import json
import random

def time_bins_and_dataset_split(input_file, num_bins, save_dir, seed, type_ = "normal", dataset = "wo", setting = "internal"):

    with open(input_file, 'r') as f:
        data = json.load(f)

    records = []
    for entry in data:
        for key, value in entry.items():
            key = key.split('_____')[0] if "_____" in key else key
            sample_name = key
            survival_time = float(value["survival_time"])
            c = value["c"]
            records.append([sample_name, survival_time, c])

    df = pd.DataFrame(records, columns=["sample_name", "survival_time", "c"])
    df_c1 = df[df["c"] == '0'].sort_values(by="survival_time")

    max_time = df_c1["survival_time"].max()
    min_time = df_c1["survival_time"].min()

    interval = (max_time - min_time) / num_bins
    bins = [min_time + interval * i for i in range(num_bins + 1)]

    df_c0 = df[df["c"] == '1']

    def assign_label(survival_time, is_c1):
        if is_c1:
            for i in range(len(bins) - 1):
                if survival_time <= bins[i + 1]:
                    return i
        else:
            for i in range(len(bins) - 1):
                if survival_time <= bins[i + 1]:
                    return i

        return len(bins) - 2

    df_c1['label'] = df_c1['survival_time'].apply(assign_label, is_c1=True)

    df_c0_copy = df_c0.copy()
    df_c0_copy['label'] = df_c0_copy['survival_time'].apply(assign_label, is_c1=False)

    df_combined = pd.concat([df_c1, df_c0_copy])

    for entry in data:
        for key, value in entry.items():
            key = key.split('_____')[0] if "_____" in key else key
            matching_rows = df_combined[df_combined["sample_name"] == key]
            if not matching_rows.empty:
                matching_row = matching_rows.iloc[0]
                value["label"] = int(matching_row["label"])
            else:
                print(f"No match found for sample_name: {key}")
                value["label"] = None

    img_split_root = "finetune_survival/features/public_split"

    if setting == "internal" and dataset != "wo":
        img_split_dir = f"{img_split_root}/img_split_{seed}_{dataset}.json"
        with open(img_split_dir, 'r') as f:
            img_split = json.load(f)

    elif setting == "external":
        img_split = {"train_img_names": [], "val_img_names": [], "test_img_names": []}
        for dataset_ in ["anzhen", "renji", "tongji", "gulou"]:
            if dataset_ != dataset:
                with open(f"{img_split_root}/img_split_{seed}_{dataset_}.json", 'r') as f:
                    img_split_dataset = json.load(f)
                    for split_type in img_split:
                        img_split[split_type].extend(img_split_dataset[split_type])

    else:
        print("cohort not specified")
        img_split_dir = f"{img_split_root}/Img_split.json"
        with open(img_split_dir, 'r') as f:
            img_split = json.load(f)

    train_img_names = img_split["train_img_names"]
    val_img_names = img_split["val_img_names"]
    test_img_names = img_split["test_img_names"]

    samples = []
    for entry in data:
        for sample_name, sample_data in entry.items():
            sample_name = sample_name.split('_____')[0] if "_____" in sample_name else sample_name
            samples.append({sample_name: sample_data})

    random.seed(seed)
    random.shuffle(samples)
    train_samples, val_samples, test_samples = [], [], []
    for sample in samples:
        for k, v in sample.items():
            if k in train_img_names:
                train_samples.append(sample)
            elif k in test_img_names:
                test_samples.append(sample)
            elif k in val_img_names:
                val_samples.append(sample)

    if type_ == "normal":
        final_train_data = train_samples
    elif type_ == "feature":
        final_train_data = train_samples + val_samples + test_samples
    elif type_ == "ssl_pretrain":
        final_train_data = train_samples + test_samples

    final_val_data = val_samples
    final_test_data = test_samples

    print("test_data_len:", len(final_test_data))

    output_data = {
        "train": final_train_data,
        "val": final_val_data,
        "test": final_test_data
    }

    output_file = f'{save_dir}/processed_and_split_data.json' if type_  == "normal" else f'{save_dir}/all_data_in_train.json'
    with open(output_file, 'w', encoding='utf-8') as f2:
        json.dump(output_data, f2, indent=3, ensure_ascii=False)

    print(f"Processed and split results saved to {output_file}")
    return output_file
