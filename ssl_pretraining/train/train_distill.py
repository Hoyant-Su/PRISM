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
from models.backbone.DINO_pretrain.DINO_distill import DINODistill

from dataloader.prognosis_4D_loader import PrognosisDataset
from dataloader.equal_time_bins_split import time_bins_and_dataset_split
from dataloader.data_structure import CineModalInput, cine_collate_fn

from models.backbone.compare_models.uniformer import uniformer_base_IL
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
from visual_utils.cam import get_attention_map_from_layer_outputs, save_video_as_images
from utils.attention_rollout import AttentionVisualizer
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

def load_backbone_weights(model, pretrained_backbone_path, load_pretrained):

    checkpoint = torch.load(pretrained_backbone_path, map_location='cpu')

    model_dict = model.state_dict()
    print(model_dict.keys())

    checkpoint_dict = checkpoint
    new_state_dict = {}

    removed_keys = ['norm.weight', 'norm.bias', 'norm.running_mean', 'norm.running_var', 'norm.num_batches_tracked', 'head.weight', 'head.bias']
    for key, value in checkpoint_dict.items():
        if 'patch_embed1.proj.bias' not in key:

            if key.startswith("module."):
                new_key = key[7:]
            else:
                new_key = key

        new_state_dict[new_key] = value

    model_dict.update(new_state_dict)
    if load_pretrained:
        model.load_state_dict(model_dict)

    print("Backbone weights loaded successfully!")
    return model

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
        img_2ch, img_3ch, img_4ch = input.img_2ch, input.img_3ch, input.img_4ch
        temporal_optical_img, CH_temporal_img_optical_ch2, CH_temporal_img_optical_ch3, CH_temporal_img_optical_ch4 = input.temporal_optical_img, input.CH_temporal_img_optical_ch2, input.CH_temporal_img_optical_ch3, input.CH_temporal_img_optical_ch4

        distill_loss = model(temporal_img, img_2ch, img_3ch, img_4ch)

        loss = distill_loss

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

    visualizer = AttentionVisualizer()
    viz_save_dir = os.path.join(log_dir, "attention_rollout", f"epoch_{epoch}")
    os.makedirs(viz_save_dir, exist_ok=True)

    visualization_done_for_epoch = False

    total_loss,  total_samples = 0.0, 0
    dataloader_tqdm = tqdm(dataloader, desc="Validation", unit="batch")

    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img
        img_2ch, img_3ch, img_4ch = input.img_2ch, input.img_3ch, input.img_4ch
        temporal_optical_img, CH_temporal_img_optical_ch2, CH_temporal_img_optical_ch3, CH_temporal_img_optical_ch4 = input.temporal_optical_img, input.CH_temporal_img_optical_ch2, input.CH_temporal_img_optical_ch3, input.CH_temporal_img_optical_ch4

        current_batch_forward_args = (
            temporal_img, img_2ch, img_3ch, img_4ch,

        )
        current_batch_forward_kwargs = {
            'teacher_temp': cfg["teacher_temp"],
            'step': idx,
            'epoch': epoch
        }

        distill_loss = model(temporal_img, img_2ch, img_3ch, img_4ch, step=idx, epoch=epoch)

        loss = distill_loss

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

    avg_loss = total_loss / total_samples

    save_path = os.path.join(f"{log_dir}", "results", cfg["date"], "distill")
    os.makedirs(save_path, exist_ok=True)
    csv_file = f"{save_path}/distill_result.csv"

    metrics = {
        "epoch": epoch,
        "total_loss": avg_loss
    }
    fieldnames = ["epoch", "total_loss"]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
        print(f"Metrics saved to {csv_file}")

    os.makedirs(os.path.join(save_path, "cam"), exist_ok=True)

    return avg_loss

def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "date": "0527_attention_viz",
        "model": "uniformer_base_IL",
        "num_time_bins": 4,
        "embed_dim": [64, 128, 320, 512],
        "in_chans_student": 24,
        "in_chans_teacher": 1,
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
        "seed": 37,
        "apply_spatial_sampling": False,
        "teacher_temp": 0.1,
        "center_momentum": 0.2
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

    save_path = os.path.join(work_space, "results", cfg["date"], "distill")
    os.makedirs(save_path, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = os.path.join(save_path, current_date)
    os.makedirs(log_dir, exist_ok=True)

    cfg["json_file"] = '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json'
    cfg["json_file"] = time_bins_and_dataset_split(cfg["json_file"], cfg["num_time_bins"], log_dir, cfg["seed"], type_="ssl_pretrain")

    if cfg["apply_spatial_sampling"]:
        cfg["in_chans_student"] = 3

    model = DINODistill(cfg).to(device)

    print(model)

    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    print(f"Total parameters: {total_params}")

    train_dataset = PrognosisDataset(cfg["json_file"], "train",  cfg["in_chans_student"], cfg["frames"], cfg["model"], cfg["use_crop"], crop_size=cfg["crop_size"],  transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"], apply_spatial_sampling=cfg["apply_spatial_sampling"])
    val_dataset = PrognosisDataset(cfg["json_file"],  "val", cfg["in_chans_student"], cfg["frames"], cfg["model"], cfg["use_crop"],  crop_size=cfg["crop_size"], transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"], apply_spatial_sampling=cfg["apply_spatial_sampling"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=16, collate_fn=cine_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=16, collate_fn=cine_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0)

    num_epochs = 200

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        result_csv_path = os.path.join(log_dir, "loss_acc.csv")

        if val_loss < best_loss:
            best_loss = val_loss
            for file in glob.glob(os.path.join(log_dir, "best_distill_epoch_*.pth")):
                os.remove(file)
            torch.save(model.state_dict(), os.path.join(f"{log_dir}", f"best_distill_epoch_{epoch}.pth"))
        torch.save(model.state_dict(), os.path.join(f"{log_dir}", f"best_last_distill.pth"))

if __name__ == "__main__":
    main()
