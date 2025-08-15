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
from sklearn.linear_model import LassoCV
import numpy as np
import yaml
import shutil
import time
from omegaconf import OmegaConf

yaml_config_path = "finetune_survival/CTSL_Cox/configs/CTSL_IECV.yaml"

cfg = OmegaConf.load(yaml_config_path)
cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
cfg = OmegaConf.merge(cfg, cli_cfg)

work_space = cfg.base_config.work_space
public_feature_processing_root = cfg.base_config.public_feature_processing_root

import sys
sys.path.append(public_feature_processing_root)
from utils.dl_based_df_split_IECV import proceed, proceed_internal
from utils.metric_visual import plot_kaplan_meier
from utils.dl_based_df_split import clean_and_split

class CTSL_Cox:
    def __init__(self, penalizer):
        self.cph = CoxPHFitter(penalizer=penalizer)

    def fit(self, df, duration_col, event_col, strata_col=None, feature_selection_method=None, force_include=None):
        if force_include is None:
            force_include = []

        df_features = df.drop(df.columns[0], axis=1)
        drop_cols = [duration_col, event_col]
        if strata_col:
            drop_cols.append(strata_col)
        df_features = df_features.drop(columns=drop_cols, errors='ignore')

        forced_features = [f for f in force_include if f in df_features.columns]
        selectable_features = [f for f in df_features.columns if f not in forced_features]
        if feature_selection_method == 'lasso':
            selected_features = self.lasso_feature_selection(df[selectable_features], df[duration_col], df[event_col])
        elif feature_selection_method == 'stepwise':
            selected_features = self.stepwise_selection(df[selectable_features], df[duration_col], df[event_col])
        else:
            selected_features = df[selectable_features].columns.tolist()

        final_features = forced_features + selected_features

        fit_df = df[final_features + [duration_col, event_col]]
        if strata_col:
            fit_df[strata_col] = df[strata_col]

            self.cph.fit(fit_df, duration_col=duration_col, event_col=event_col, strata=strata_col)
        else:
            self.cph.fit(fit_df, duration_col=duration_col, event_col=event_col)
        self.selected_columns = final_features

        return self.cph

    def stepwise_selection(self, X, y, event, direction='both', threshold_in=0.01, threshold_out=0.05):
        initial_features = X.columns.tolist()
        selected_features = []

        while True:
            changed = False

            if direction in ['both', 'forward']:
                new_features = list(set(initial_features) - set(selected_features))
                p_values = []
                for feature in new_features:
                    X_new = X[selected_features + [feature]]
                    df_new = pd.concat([X_new, y, event], axis=1)
                    self.cph.fit(df_new, duration_col=y.name, event_col=event.name)
                    p_values.append(self.cph.summary.loc[feature, "p"])

                min_p_value = min(p_values)
                if min_p_value < threshold_in:
                    selected_features.append(new_features[p_values.index(min_p_value)])
                    changed = True

            if direction in ['both', 'backward']:
                for feature in selected_features.copy():
                    X_new = X[selected_features].drop(columns=feature)
                    df_new = pd.concat([X_new, y, event], axis=1)
                    self.cph.fit(df_new, duration_col=y.name, event_col=event.name)

                    if feature not in self.cph.summary.index:
                        continue
                    p_value = self.cph.summary.loc[feature, "p"]
                    if p_value > threshold_out:
                        selected_features.remove(feature)
                        changed = True

            if not changed:
                break

        return selected_features

    def lasso_feature_selection(self, X, y, event_col):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lasso = LassoCV(cv=5)
        lasso.fit(X_train, y_train)

        selected_features = X.columns[lasso.coef_ != 0]
        return selected_features.tolist()

    def print_summary(self):
        buffer = io.StringIO()
        sys.stdout = buffer
        self.cph.print_summary()
        summary = buffer.getvalue()
        sys.stdout = sys.__stdout__
        return summary

    def predict_risk(self, df):
        df = df.copy()
        df_features = df[self.selected_columns].copy()
        df['risk_score'] = self.cph.predict_partial_hazard(df_features)
        return df

    def save_model(self, file_name):

        self.cph.save(file_name)

if __name__ == "__main__":

    dataset = cfg.data.dataset
    apply_strata = cfg.tricks.apply_strata
    apply_stepwise = cfg.tricks.apply_stepwise
    trick_types = cfg.tricks.feature_selection_method
    dl_csv = cfg.data.dl_csv
    model = cfg.model.name
    seed = cfg.hyperparam.seed
    inference_prompt = cfg.data.inference_prompt

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{work_space}/logs/{model}/{inference_prompt}/{cfg.setting}_{cfg.data.ehr_group}/seed_{seed}/{dataset}_{current_time}_strata_{apply_strata}_{trick_types}_{apply_stepwise}_fix_Img'

    if not os.path.exists(f"{log_dir}"):
        os.makedirs(f"{log_dir}")

    log_file = os.path.join(log_dir, f'{model}_cox_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    t_p_dict = {"renji":[0.7, 0.0001], "anzhen":[0.9, 0.01], "tongji":[0.7, 0.01]}

    shutil.copy(yaml_config_path, f"{log_dir}/config.yaml")

    threshold_range = cfg.hyperparam.threshold_search_range
    penalizer_range = cfg.hyperparam.penalizer_search_range

    c_index_list = []
    avg_train_c_index_list = []
    auc_list = []

    ehr_group = cfg.data.ehr_group
    for threshold in threshold_range:
        for penalizer in penalizer_range:
            if cfg.setting == "external":
                train_df, val_df, test_df, _ = proceed(dataset, dl_csv, ehr_group, threshold=threshold, apply_strata = apply_strata, seed=seed)

                _, _, test_df_ref = proceed_internal(dataset, dl_csv, ehr_group, threshold=threshold, apply_strata = apply_strata, seed=seed)
                reference_img_names = set(test_df_ref['img_name'])
                mask = test_df['img_name'].isin(reference_img_names)
                test_df = test_df[mask]
            else:
                train_df, val_df, test_df = proceed_internal(dataset, dl_csv, ehr_group, threshold=threshold, apply_strata = apply_strata, seed=seed)

            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            cox_model = CTSL_Cox(penalizer=penalizer)

            print(len(train_val_df))
            print(len(test_df))

            start_time = time.time()

            if cfg.tricks.fix_img:
                force_include = [col for col in train_val_df.columns if 'feat_' in col]
            else:
                force_include = []

            if apply_strata and apply_stepwise:
                cox_model.fit(
                    train_val_df, duration_col='survival_time', event_col='Mace', strata_col='dataset_name', feature_selection_method=cfg.tricks.feature_selection_method,
                    force_include=force_include
                )
            elif apply_strata and not apply_stepwise:
                cox_model.fit(
                    train_val_df, duration_col='survival_time', event_col='Mace', strata_col='dataset_name',
                    force_include=force_include
                )
            elif not apply_strata and apply_stepwise:
                cox_model.fit(
                    train_val_df,duration_col='survival_time',event_col='Mace', feature_selection_method=cfg.tricks.feature_selection_method,
                    force_include=force_include
                )
            else:
                cox_model.fit(
                    train_val_df, duration_col='survival_time', event_col='Mace',
                    force_include=force_include
                )
            end_time = time.time()
            print("training time:", end_time - start_time)

            logging.info(f"Threshold for multicollinearity: {threshold}")
            summary = cox_model.print_summary()
            logging.info(summary)

            train_risk_df = cox_model.predict_risk(train_df)
            train_c_index = concordance_index_censored(train_df['Mace'].astype(bool), train_df['survival_time'], train_risk_df['risk_score'])
            logging.info(f"Train C-Index: {train_c_index}")

            start_time = time.time()
            test_risk_df = cox_model.predict_risk(test_df)
            end_time = time.time()
            print("testing time:", end_time - start_time)

            print(test_risk_df["risk_score"])

            max_risk_value = 1e10
            test_risk_df['risk_score'] = np.where(test_risk_df['risk_score'] > max_risk_value, max_risk_value, test_risk_df['risk_score'])

            test_c_index = concordance_index_censored(test_df['Mace'].astype(bool), test_df['survival_time'], test_risk_df['risk_score'])
            logging.info(f"Penalizer:{penalizer}")
            logging.info(f"Test C-Index: {test_c_index}")

            plot_kaplan_meier(f"{model}", test_risk_df['risk_score'].to_numpy(), test_df['survival_time'].to_numpy(), test_df['Mace'].to_numpy(), log_dir, test_c_index, threshold=threshold, penalizer=penalizer)

            test_auc = roc_auc_score(test_df['Mace'], test_risk_df['risk_score'])
            logging.info(f"Test AUC: {test_auc}")
            logging.info("=" * 50)

            avg_train_c_index_list.append(train_c_index)
            c_index_list.append(test_c_index)
            auc_list.append(test_auc)

    max_c_index_idx = c_index_list.index(max(c_index_list))

    logging.info(f"Max C-index & AUC: {c_index_list[max_c_index_idx]}, {auc_list[max_c_index_idx]}")
