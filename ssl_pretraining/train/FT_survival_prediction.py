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
from models.backbone.text_encoder import Rad_TextEncoder

from models.backbone.compare_models.uniformer import uniformer_base_IL
from models.backbone.compare_models.uniformer_optical_included import OpticalPromptedUniFormer
from models.backbone.compare_models.Med3D import resnet200
from models.backbone.compare_models.Clinical_NN import ClinicalDataNN
from models.backbone.compare_models.Sparse_bagnet.surv_cnn import create_survival_sparsebagnet33

from dataloader.prognosis_4D_loader import PrognosisDataset

from dataloader.equal_time_bins_split import time_bins_and_dataset_split
from dataloader.data_structure import CineModalInput, cine_collate_fn

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import argparse

from utils.survival_loss import NLLSurvLoss_dep, CrossEntropySurvLoss, compute_c_index_with_threshold, compute_c_index_with_diff, compute_c_index_with_risk
from sksurv.metrics import concordance_index_censored
from utils.metric_visual import plot_kaplan_meier
from utils.brier_score import calc_brier_score
from utils.global_norm import calculate_global_mean_std, normalize_data
from utils.pesudo_survival_tail import generate_pseudo_labels, pesudo_sigmoid, generate_pseudo_labels_with_decay
from utils.pesudo_learning_module import TruncationModule
from utils.backbone_weight_loader import load_backbone_weights

def set_seed(seed):
    import random, torch, numpy as np, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_weights(dataset):

    labels = [sample[1] for sample in dataset]
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in labels])

    return sample_weights

def setup_ddp(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():

    dist.destroy_process_group()

def train_one_epoch(model, dataloader, criterion, optimizer, device, cfg, pesudo_learning_module, epoch_idx):
    model.train()
    total_loss, total_samples = 0.0, 0
    l2_lambda =  5e-6

    num_time_bins = cfg["num_time_bins"]
    batch_size = cfg["batch_size"]
    model_name = cfg["model_student"]
    dataloader_tqdm = tqdm(dataloader, desc="Training", unit="batch")

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
        elif model_name == 'clinical_nn':
            outputs = model(clinical_indicators)
        elif model_name == 'RadBert':
            outputs = model(text)

        elif model_name == "SparseBagnet":
            B, D, T, H, W = temporal_img.shape
            temporal_img, spatial_img = temporal_img[:,D//2,T//2:T//2+3,:,:], spatial_img[:,D//2:D//2+3,T//2,:,:]
            outputs, _ = model(torch.cat([temporal_img.unsqueeze(1), spatial_img.unsqueeze(1)], dim=1))
            outputs = torch.mean(outputs, dim=1)
        elif model_name == 'OpticalPromptedUniFormer':
            outputs = model(temporal_img, temporal_optical_img)
        else:
            outputs = model(temporal_img)
        sigmoid = nn.Sigmoid()
        outputs = sigmoid(outputs)

        S = torch.cumprod(1 - outputs, dim=1)

        if not cfg["use_pesudo_label"] or epoch_idx < 30:
            loss = criterion(outputs, None, labels_Y, labels_c) if model_name != "vqvae" else (criterion(outputs, None, labels_Y, labels_c) + criterion(inputs_feat, None, labels_Y, labels_c)) / 2

        if cfg["use_pesudo_label"] and epoch_idx >= 30:

            truncation_pos, S, pesudo_loss = pesudo_learning_module(outputs, labels_Y)
            loss = criterion(outputs, S, labels_Y, labels_c)
            w1, w2 = (pesudo_loss) / (pesudo_loss + loss), loss / (pesudo_loss + loss)
            loss  = w1 * loss + w2 * pesudo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

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

    brier_score = calc_brier_score(all_survival_preds, all_labels_c, all_labels_Y, batch_size, num_time_bins)
    c_index = concordance_index_censored((1-all_labels_c).astype(bool), all_labels_time, all_preds, tied_tol=1e-08)[0]

    avg_loss = total_loss / total_samples

    print(f"C-Index: {c_index:.4f}; Brier Score: {brier_score}")

    return avg_loss, c_index

@torch.no_grad()
def validate(model, dataloader, criterion, device, cfg, epoch, save_path, pesudo_learning_module, epoch_idx):
    model.eval()

    total_loss, total_samples = 0.0, 0

    num_time_bins = cfg["num_time_bins"]
    batch_size = cfg["batch_size"]
    model_name = cfg["model_student"]

    dataloader_tqdm = tqdm(dataloader, desc="Validation", unit="batch")

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
        elif model_name == 'clinical_nn':
            outputs = model(clinical_indicators)
        elif model_name == 'RadBert':
            outputs = model(text)
        elif model_name == "SparseBagnet":
            B, D, T, H, W = temporal_img.shape
            temporal_img, spatial_img = temporal_img[:,D//2,T//2:T//2+3,:,:], spatial_img[:,D//2:D//2+3,T//2,:,:]
            outputs, _ = model(torch.cat([temporal_img.unsqueeze(1), spatial_img.unsqueeze(1)], dim=1))
            outputs = torch.mean(outputs, dim=1)
        elif model_name == 'OpticalPromptedUniFormer':
            outputs = model(temporal_img, temporal_optical_img)
        else:
            outputs = model(temporal_img)
        sigmoid = nn.Sigmoid()
        outputs = sigmoid(outputs)

        loss = criterion(outputs, None, labels_Y, labels_c)

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

        S = torch.cumprod(1 - outputs, dim=1)

        if not cfg["use_pesudo_label"] or epoch_idx < 30:
            loss = criterion(outputs, None, labels_Y, labels_c) if model_name != "vqvae" else (criterion(outputs, None, labels_Y, labels_c) + criterion(inputs_feat, None, labels_Y, labels_c)) / 2

        if cfg["use_pesudo_label"] and epoch_idx >= 30:

            truncation_pos, S, pesudo_loss = pesudo_learning_module(outputs, labels_Y)
            loss = criterion(outputs, S, labels_Y, labels_c)
            w1, w2 = (pesudo_loss) / (pesudo_loss + loss), loss / (pesudo_loss + loss)
            loss  = w1 * loss + w2 * pesudo_loss

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

    print(result)
    avg_loss = total_loss / total_samples

    print(f"C-Index: {c_index:.4f}; Brier Score: {brier_score}")

    csv_file = f'{save_path}/validation_metrics.csv'
    fieldnames = ['Epoch', 'validation_loss', 'CI', 'brier_score']

    metrics = {
        'Epoch': epoch,
        'validation_loss': avg_loss,
        'CI': c_index,
        'brier_score': brier_score
    }

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
        print(f"Metrics saved to {csv_file}")

    return avg_loss, c_index

def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
        "datasets": 'tongji',
        "phase": 'train',
        "crop_size": 96,
        "batch_size": 16,
        "lr": 5e-5,
        "cls_label_type": "survival_time",
        "load_pretrained_weight": False,
        "frozen_backbone": False,
        "use_crop": True,
        "use_asymmetric": True,
        "use_nll": True,
        "ones_focus": False,
        "alpha": 0.7,
        "use_augment": True,
        "variable_stride": (2, 4, 4),
        "equal_time_split": True,
        "seed": 57131,
        "use_distill_model": True,
        "use_pesudo_label": True,
        "use_feature_output": False,
        "setting": "external"
    }

    set_seed(cfg["seed"])
    save_path = os.path.join(cfg["work_space"], "results", cfg["date"], "survival_prediction")
    os.makedirs(save_path, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = os.path.join(save_path, f"{current_date}")
    os.makedirs(log_dir, exist_ok=True)

    cfg["json_file"] = '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json'
    cfg["json_file"] = time_bins_and_dataset_split(cfg["json_file"], cfg["num_time_bins"], log_dir, cfg["seed"], dataset=cfg["datasets"], setting=cfg["setting"])

    if cfg["model_student"] == "uniformer_base_IL":

        model = uniformer_base_IL(
            in_chans = cfg["depth"],

            num_classes = cfg["num_time_bins"],
            variable_stride = cfg["variable_stride"]

        )

        if cfg["use_distill_model"]:

            model = load_backbone_weights(model, pretrained_backbone_path, cfg["load_pretrained_weight"], model_name = "uniformer").to(device)

    elif cfg["model_student"] == "vqvae":
        model = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans_temporal=cfg["depth"], in_chans_spatial=cfg["frames"], variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05).to(device)
        pretrained_backbone_path = f"{work_space}/results/0218_vqvae/codebook/2025-02-18 04:50:58/best_codebook_epoch_13.pth"
        model_backbone = load_backbone_weights(model, pretrained_backbone_path, cfg["load_pretrained_weight"]).to(device)

        survival_prediction_head = Cine_Survival_Prediction_Head(cfg, model_backbone).to(device)
        model = survival_prediction_head

        for name, param in model.named_parameters():

            if 'decoder' in name:
                param.requires_grad = False

    elif cfg["model_student"] == "OpticalPromptedUniFormer":
        model = OpticalPromptedUniFormer(num_classes = cfg["num_time_bins"],variable_stride = cfg["variable_stride"], in_chans=cfg["depth"])

    elif cfg["model_student"] == "Med3D":
        model = resnet200(num_classes=cfg["num_time_bins"])

    elif cfg["model_student"] == "clinical_nn":
        model = ClinicalDataNN(input_dim=8, num_time_bins=cfg["num_time_bins"])

    elif cfg["model_student"] == 'RadBert':
        model = Rad_TextEncoder(num_classes = cfg["num_time_bins"], frozen_text_encoder=False, pretrained_model_path='../TextEncoder/BERT/bert_model')

    elif cfg["model_student"] == 'SparseBagnet':
        model = create_survival_sparsebagnet33(num_times = cfg["num_time_bins"])

    model = model.to(device)
    pesudo_learning_module = TruncationModule(num_time_bins= cfg["num_time_bins"]).to(device)

    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params}; Trainable params: {trainable_params}')

    train_dataset = PrognosisDataset(cfg["json_file"], "train",  cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"], crop_size=cfg["crop_size"],  transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"])
    val_dataset = PrognosisDataset(cfg["json_file"],  "val", cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"],  crop_size=cfg["crop_size"], transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=cine_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, collate_fn=cine_collate_fn)

    if cfg["use_nll"]:
        criterion_train = NLLSurvLoss_dep(alpha=cfg["alpha"]).to(device)
        criterion_val = NLLSurvLoss_dep(alpha=cfg["alpha"]).to(device)
    else:
        criterion_train = CrossEntropySurvLoss(alpha=0.15).to(device)
        criterion_val = CrossEntropySurvLoss(alpha=0.15).to(device)

    num_epochs = 20

    if not cfg["use_pesudo_label"]:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
    else:
        optimizer = optim.Adam(
            list(model.parameters()) + list(pesudo_learning_module.parameters()),
            lr=cfg["lr"],
            weight_decay=0
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_CI = 0

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

    for epoch_idx, epoch in enumerate(range(num_epochs)):
        train_loss, train_CI = train_one_epoch(model, train_loader, criterion_train, optimizer, device, cfg, pesudo_learning_module, epoch_idx)
        val_loss, val_CI = validate(model, val_loader, criterion_val, device, cfg, epoch, log_dir, pesudo_learning_module, epoch_idx)

        scheduler.step()

        result_csv_path = os.path.join(log_dir, "loss_acc.csv")
        with open(result_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            writer.writerow([epoch + 1, float(f"{train_loss:.4f}"), float(f"{val_loss:.4f}")])

        if val_CI > best_val_CI:
            if best_val_CI > 0:
                for file in glob.glob(os.path.join(log_dir, "best_*.pth")):
                    os.remove(file)

            best_val_CI = val_CI
            best_CI_epoch = epoch
            torch.save(model.state_dict(), os.path.join(f"{log_dir}", f"best_CI_epoch_{best_CI_epoch}.pth"))
            torch.save(pesudo_learning_module.state_dict(), os.path.join(f"{log_dir}", f"best_pesudo_module_epoch_{best_CI_epoch}.pth"))

        torch.save(model.state_dict(), os.path.join(f"{log_dir}", f"best_last_CI_epoch.pth"))

if __name__ == "__main__":
    main()
