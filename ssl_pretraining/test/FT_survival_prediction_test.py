import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import logging
import glob

import csv
import pdb
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

work_space = 'ssl_pretraining'
sys.path.append(work_space)
sys.path.append(f"{work_space}/dataloader")

from models.backbone.Recon_vqvae import VQ_VAE_4D
from models.head.survival_prediction import Cine_Survival_Prediction_Head

from models.backbone.compare_models.uniformer import uniformer_base_IL
from models.backbone.compare_models.Med3D import resnet200
from models.backbone.compare_models.Sparse_bagnet.surv_cnn import create_survival_sparsebagnet33

from dataloader.prognosis_4D_loader import PrognosisDataset

from dataloader.equal_time_bins_split import time_bins_and_dataset_split
from dataloader.data_structure import CineModalInput, cine_collate_fn

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import argparse

from utils.survival_loss import NLLSurvLoss_dep, CrossEntropySurvLoss, compute_c_index_with_threshold, compute_c_index_with_diff, compute_c_index_with_risk
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score
from utils.metric_visual import plot_kaplan_meier
from utils.brier_score import calc_brier_score
from utils.pesudo_learning_module import TruncationModule

def calculate_weights(dataset):
    labels = [sample[1] for sample in dataset]
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in labels])

    return sample_weights

def load_weights(model, model_weights):
    checkpoint = torch.load(model_weights, map_location='cpu')
    model_dict = model.state_dict()
    checkpoint_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)

    print("Pretrained weights loaded successfully!")
    return model

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

@torch.no_grad()
def test(model, dataloader, criterion, device, cfg, save_path, pesudo_learning_module, dataset=None):
    model.eval()

    total_loss, total_samples = 0.0, 0

    num_time_bins = cfg["num_time_bins"]
    batch_size = cfg["batch_size"]
    model_name = cfg["model_student"]

    dataloader_tqdm = tqdm(dataloader, desc="Test", unit="batch")

    all_preds = []
    all_survival_preds = []
    all_labels_Y = []
    all_labels_c = []
    all_labels_time = []

    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img
        img_2ch, img_3ch, img_4ch = input.img_2ch, input.img_3ch, input.img_4ch
        temporal_optical_img, CH_temporal_img_optical_ch2, CH_temporal_img_optical_ch3, CH_temporal_img_optical_ch4 = input.temporal_optical_img, input.CH_temporal_img_optical_ch2, input.CH_temporal_img_optical_ch3, input.CH_temporal_img_optical_ch4
        labels_c, labels_Y, labels_time = input.labels_c, input.labels_Y, input.labels_time
        text, prompt_text = input.text, input.prompt_text
        clinical_indicators = input.clinical_indicators

        if model_name == 'vqvae':
            outputs, inputs_feat = model(temporal_img, spatial_img)
        elif model_name == "SparseBagnet":
            B, D, T, H, W = temporal_img.shape
            temporal_img, spatial_img = temporal_img[:,D//2,T//2:T//2+3,:,:], spatial_img[:,D//2:D//2+3,T//2,:,:]
            outputs, _ = model(torch.cat([temporal_img.unsqueeze(1), spatial_img.unsqueeze(1)], dim=1))
            outputs = torch.mean(outputs, dim=1)
        else:
            outputs = model(temporal_img)
        sigmoid = nn.Sigmoid()
        outputs = sigmoid(outputs)

        loss = criterion(outputs, None, labels_Y, labels_c)

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

        S = torch.cumprod(1 - outputs, dim=1)

        if cfg["use_pesudo_label"]:
            _, S, _ = pesudo_learning_module(outputs, labels_Y)

        print(outputs)
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()

        all_preds.append(risk)
        all_survival_preds.append(S.detach().cpu().numpy())
        all_labels_time.append(labels_time.detach().cpu().numpy())
        all_labels_c.append(labels_c.detach().cpu().numpy())
        all_labels_Y.append(labels_Y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_survival_preds = np.concatenate(all_survival_preds, axis=0)
    all_labels_time = np.concatenate(all_labels_time, axis=0)
    all_labels_c = np.concatenate(all_labels_c, axis=0)
    all_labels_Y = np.concatenate(all_labels_Y, axis=0)

    result = concordance_index_censored((1-all_labels_c).astype(bool), all_labels_time, all_preds, tied_tol=1e-08)
    c_index = result[0]
    brier_score = calc_brier_score(all_survival_preds, all_labels_c, all_labels_Y, batch_size, num_time_bins)
    test_auc = 1 - roc_auc_score(all_labels_c, all_preds)

    print(result)
    avg_loss = total_loss / total_samples

    print(f"C-Index: {c_index:.4f}; Brier Score: {brier_score}")

    os.makedirs(f'{save_path}/{dataset}_seed{cfg["seed"]}', exist_ok=True)
    csv_file = f'{save_path}/{dataset}_seed{cfg["seed"]}/test_metrics.csv'
    fieldnames = ['test_loss', 'CI', 'AUC', 'brier_score']

    metrics = {
        'test_loss': avg_loss,
        'CI': c_index,
        'AUC': test_auc,
        'brier_score': brier_score
    }

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
        print(f"Metrics saved to {csv_file}")

    plot_kaplan_meier(model_name, all_preds, all_labels_time, all_labels_c, save_path, c_index, cfg=cfg, dataset=dataset)

    return avg_loss, c_index, test_auc

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "date": "0617_SparseBagnet",
        "model_student": "SparseBagnet",
        "num_time_bins": 4,
        "embed_dim": [64, 128, 320, 512],
        "depth": 24,
        "frames": 24,
        "work_space": work_space,
        "json_file": '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json',
        "datasets": 'gulou',
        "phase": 'train',
        "crop_size": 96,
        "batch_size": 8,
        "cls_label_type": "survival_time",
        "frozen_backbone": False,
        "use_crop": True,
        "use_asymmetric": True,
        "use_nll": True,
        "ones_focus": False,
        "alpha": 0.7,
        "use_augment": True,
        "variable_stride": (2, 4, 4),
        "equal_time_split": True,
        "seed": 256,
        "use_distill_model": True,
        "use_pesudo_label": False,
        "use_feature_output": False
    }

    trained_model_path = "ssl_pretraining/results/0617_SparseBagnet/survival_prediction/2025-06-20 09:16:49/best_CI_epoch_3.pth"
    trained_pesudo_path = trained_model_path.replace('CI', 'pesudo_module')

    save_path = os.path.join(cfg["work_space"], "results", cfg["date"], "survival_prediction")
    os.makedirs(save_path, exist_ok=True)

    log_dir = trained_model_path.split('/best')[0]
    os.makedirs(log_dir, exist_ok=True)

    cfg["json_file"] = '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json'
    cfg["json_file"] = time_bins_and_dataset_split(cfg["json_file"], cfg["num_time_bins"], log_dir, cfg["seed"], dataset=cfg["datasets"])

    if cfg["model_student"] == "uniformer_base_IL":

        model = uniformer_base_IL(
            in_chans = cfg["depth"],

            num_classes = cfg["num_time_bins"],
            variable_stride = cfg["variable_stride"]

        )

    elif cfg["model_student"] == "vqvae":
        model_backbone = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans_temporal=cfg["depth"], in_chans_spatial=cfg["frames"], variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05).to(device)
        survival_prediction_head = Cine_Survival_Prediction_Head(cfg, model_backbone).to(device)
        model = survival_prediction_head

    elif cfg["model_student"] == "Med3D":
        model = resnet200(num_classes=cfg["num_time_bins"])

    elif cfg["model_student"] == 'SparseBagnet':
        model = create_survival_sparsebagnet33(num_times = cfg["num_time_bins"])

    model = load_weights(model, trained_model_path).to(device)

    pesudo_learning_module = TruncationModule(num_time_bins= cfg["num_time_bins"]).to(device)
    pesudo_learning_module = load_weights(pesudo_learning_module, trained_pesudo_path).to(device)

    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total params: {total_params}; Trainable params: {trainable_params}')

    test_dataset = PrognosisDataset(cfg["json_file"],  "test", cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"],  crop_size=cfg["crop_size"], transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"], center=cfg["datasets"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=16, collate_fn=cine_collate_fn)

    if cfg["use_nll"]:
        criterion_train = NLLSurvLoss_dep(alpha=cfg["alpha"]).to(device)
        criterion_test = NLLSurvLoss_dep(alpha=cfg["alpha"]).to(device)
    else:
        criterion_train = CrossEntropySurvLoss(alpha=0.15).to(device)
        criterion_test = CrossEntropySurvLoss(alpha=0.15).to(device)

    log_file = os.path.join(log_dir, "config.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.info("Configuration: %s", cfg)

    test_loss, test_CI, test_auc = test(model, test_loader, criterion_test, device, cfg, log_dir, pesudo_learning_module, dataset=cfg["datasets"])
    logging.info(f"Test AUC: {test_auc}")

if __name__ == "__main__":
    main()
