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
work_space = 'ssl_pretraining'
sys.path.append(work_space)

from models.backbone.Recon_vqvae import VQ_VAE_4D

from dataloader.prognosis_4D_loader import PrognosisDataset
from dataloader.equal_time_bins_split import time_bins_and_dataset_split
from dataloader.data_structure import CineModalInput, cine_collate_fn
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.backbone_weight_loader import load_backbone_weights

import argparse
from PIL import Image
import time

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

def setup_ddp(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():

    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    l2_lambda =  5e-6
    dataloader_tqdm = tqdm(dataloader, desc="Training", unit="batch")

    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img

        codebook_loss, x_hat_temporal, x_hat_spatial,_,_,_,_ = model(temporal_img, spatial_img)
        recon_loss = (torch.mean((x_hat_temporal - temporal_img)**2) + torch.mean((x_hat_spatial - spatial_img)**2)) / 2

        loss = codebook_loss + recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

    avg_loss = total_loss / total_samples

    return avg_loss

@torch.no_grad()
def validate(model, dataloader, device, cfg, epoch, log_dir):
    model.eval()
    metrics = {}

    total_loss,  total_samples = 0.0, 0
    total_codebook_loss, total_recon_loss = 0.0, 0.0
    dataloader_tqdm = tqdm(dataloader, desc="Validation", unit="batch")

    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img

        codebook_loss, x_hat_temporal, x_hat_spatial, _,_,_,_ = model(temporal_img, spatial_img)
        recon_loss = (torch.mean((x_hat_temporal - temporal_img)**2) + torch.mean((x_hat_spatial - spatial_img)**2)) / 2

        loss = codebook_loss + recon_loss

        total_loss += loss.item() * temporal_img.size(0)
        total_codebook_loss += codebook_loss.item() * temporal_img.size(0)
        total_recon_loss += recon_loss.item() * temporal_img.size(0)

        total_samples += temporal_img.size(0)

    avg_loss = total_loss / total_samples
    avg_codebook_loss = total_codebook_loss / total_samples
    avg_recon_loss = total_recon_loss / total_samples

    save_path = os.path.join(f"{log_dir}", "results", cfg["date"], "codebook")

    os.makedirs(save_path, exist_ok=True)

    csv_file = f"{save_path}/codebook_result.csv"

    metrics = {
        "epoch": epoch,
        "total_loss": avg_loss,
        'codebook_loss': avg_codebook_loss,
        'recon_loss': avg_recon_loss
    }
    fieldnames = ["epoch", "total_loss", "codebook_loss", "recon_loss"]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
        print(f"Metrics saved to {csv_file}")
    return avg_loss

def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "date": "0507_attention_viz",
        "model_student": "vqvae",
        "num_time_bins": 4,
        "embed_dim": [64, 128, 320, 512],
        "depth": 24,
        "frames": 24,
        "work_space": work_space,
        "json_file": '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json',
        "datasets": 'survival_time',
        "phase": 'train',
        "crop_size": 96,
        "batch_size": 16,
        "cls_label_type": "survival_time",
        "load_pretrained_weight": True,
        "frozen_backbone": False,
        "use_crop": True,
        "use_asymmetric": True,
        "use_nll": True,
        "ones_focus": False,
        "alpha": 0.7,
        "use_augment": True,
        "variable_stride": (2, 4, 4),
        "equal_time_split": True,
        "seed": 37
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset",  type=str, default='CIFAR10')
    args = parser.parse_args()

    set_seed(cfg["seed"])

    save_path = os.path.join(work_space, "results", cfg["date"], "codebook")
    os.makedirs(save_path, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = os.path.join(save_path, current_date)
    os.makedirs(log_dir, exist_ok=True)

    cfg["json_file"] = '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json'
    cfg["json_file"] = time_bins_and_dataset_split(cfg["json_file"], cfg["num_time_bins"], log_dir, cfg["seed"], type_="ssl_pretrain")

    pretrained_backbone_path = os.path.join(work_space, "results", cfg["date"], "student_best_epoch.pth")

    if cfg["model_student"] == "vqvae":
        model = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans_temporal=cfg["frames"],in_chans_spatial=cfg["depth"], variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.25).to(device)
        print(model)

        pretrained_backbone_path = "ssl_pretraining/results/0507_attention_viz/distill/2025-05-14 07:49:45/best_last_distill.pth"
        if cfg["load_pretrained_weight"]:
            model = load_backbone_weights(model, pretrained_backbone_path, cfg["load_pretrained_weight"], model_name = "vqvae_codebook").to(device)
        if cfg["frozen_backbone"]:
            for name, params in model.named_parameters():
                if 'encoder' in name:
                    params.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    train_dataset = PrognosisDataset(cfg["json_file"], "train",  cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"], crop_size=cfg["crop_size"],  transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"])
    val_dataset = PrognosisDataset(cfg["json_file"],  "val", cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"],  crop_size=cfg["crop_size"], transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=16, collate_fn=cine_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=16, collate_fn=cine_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 50

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file = os.path.join(log_dir, "config.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Configuration: %s", cfg)

    best_loss = 1e8

    average_time, count = 0, 0
    for epoch in range(num_epochs):

        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device, cfg, epoch, log_dir)
        end_time = time.time()
        average_time += end_time - start_time
        count += 1
        print(average_time / count)

        scheduler.step()

        result_csv_path = os.path.join(log_dir, "loss_acc.csv")

        if val_loss < best_loss:
            best_loss = val_loss
            for file in glob.glob(os.path.join(log_dir, "best_codebook_epoch_*.pth")):
                os.remove(file)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
            }, os.path.join(f"{log_dir}", f"best_codebook_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
