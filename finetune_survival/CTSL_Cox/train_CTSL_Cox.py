import pandas as pd
from lifelines import CoxPHFitter
import sys
import logging
import io
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score

work_space = 'finetune_survival/CTSL_Cox'
public_feature_processing_root = 'finetune_survival/features'
import sys
sys.path.append(public_feature_processing_root)
from utils.dl_based_df_split import clean_and_split
from utils.metric_visual import plot_kaplan_meier
from omegaconf import OmegaConf
import time

yaml_config_path = "finetune_survival/CTSL_Cox/configs/CTSL_Internal.yaml"

cfg = OmegaConf.load(yaml_config_path)
cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
cfg = OmegaConf.merge(cfg, cli_cfg)

class CTSL_Cox:
    def __init__(self, penalizer):
        self.cph = CoxPHFitter(penalizer=penalizer)

    def fit(self, df, duration_col, event_col):
        df_features = df.drop(df.columns[0], axis=1)
        self.cph.fit(df_features, duration_col=duration_col, event_col=event_col)
        return self.cph

    def print_summary(self):
        buffer = io.StringIO()
        sys.stdout = buffer
        self.cph.print_summary()
        summary = buffer.getvalue()
        sys.stdout = sys.__stdout__
        return summary

    def predict_risk(self, df):
        df_features = df.drop(df.columns[0], axis=1)

        df['risk_score'] = self.cph.predict_partial_hazard(df_features)
        return df[['img_name', 'risk_score']]

    def save_model(self, file_name):
        self.cph.save(file_name)

if __name__ == "__main__":

    dataset = cfg.data.dataset

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{work_space}/logs/{dataset}_{current_time}'

    if not os.path.exists(f"{log_dir}"):
        os.makedirs(f"{log_dir}")

    log_file = os.path.join(log_dir, 'ctsl_cox_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    t_p_dict = {"renji":[0.7, 0.0001], "anzhen":[0.9, 0.01], "tongji":[0.7, 0.01]}

apply_strata = cfg.tricks.apply_strata
apply_stepwise = cfg.tricks.apply_stepwise
trick_types = cfg.tricks.feature_selection_method
dl_csv = cfg.data.dl_csv
model = cfg.model.name
seed = cfg.hyperparam.seed

threshold_range = cfg.hyperparam.threshold_search_range
penalizer_range = cfg.hyperparam.penalizer_search_range

for threshold in threshold_range:
    for penalizer in penalizer_range:

        dl_csv = "ssl_pretraining/results/0507_attention_viz/codebook/2025-05-07 08:23:04/output_features.csv"
        model = "CTSL"

        train_df, val_df, test_df, _ = clean_and_split(work_space, dataset, dl_csv, threshold=threshold)
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)

        cox_model = CTSL_Cox(penalizer=penalizer)

        start_time = time.time()
        cox_model.fit(train_df, duration_col='survival_time', event_col='Mace')
        end_time = time.time()
        print(end_time - start_time)

        logging.info(f"Threshold for multicollinearity: {threshold}")
        summary = cox_model.print_summary()
        logging.info(summary)

        train_risk_df = cox_model.predict_risk(train_df)
        train_c_index = concordance_index_censored(train_df['Mace'].astype(bool), train_df['survival_time'], train_risk_df['risk_score'])

        start_time = time.time()

        test_risk_df = cox_model.predict_risk(test_df)
        end_time = time.time()
        print(end_time - start_time)

        print(test_risk_df["risk_score"])

        import numpy as np

        max_risk_value = 1e10
        test_risk_df['risk_score'] = np.where(test_risk_df['risk_score'] > max_risk_value, max_risk_value, test_risk_df['risk_score'])

        test_c_index = concordance_index_censored(test_df['Mace'].astype(bool), test_df['survival_time'], test_risk_df['risk_score'])
        logging.info(f"Penalizer:{penalizer}")
        logging.info(f"Test C-Index: {test_c_index}")
        logging.info("=" * 50)

        plot_kaplan_meier(f"{model}", test_risk_df['risk_score'].to_numpy(), test_df['survival_time'].to_numpy(), test_df['Mace'].to_numpy(), log_dir, test_c_index, threshold=threshold, penalizer=penalizer)

        test_auc = roc_auc_score(test_df['Mace'], test_risk_df['risk_score'])
        logging.info(f"Test AUC: {test_auc}")
