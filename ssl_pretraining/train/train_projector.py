import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import logging
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import pdb
import sys
work_space = 'ssl_pretraining'
sys.path.append(work_space)

from models.backbone.Recon_vqvae import VQ_VAE_4D

from models.head.projector_fix import DINO_Cine_Projector

from models.backbone.visual_encoder import UniformerEncoder
from models.head.routing_module import PromptMapper
from dataloader.prognosis_4D_loader_FT import PrognosisDataset
from dataloader.equal_time_bins_split import time_bins_and_dataset_split
from dataloader.data_structure import CineModalInput, cine_collate_fn
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.backbone_weight_loader import load_backbone_weights
from utils.contrastive_loss import contrastive_loss
from utils.relation_loss import RelationalDistanceLoss
import torch.multiprocessing as mp

import argparse
from PIL import Image
import signal

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

def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss, total_samples = 0.0, 0
    l2_lambda =  5e-6
    dataloader_tqdm = tqdm(dataloader, desc="Training", unit="batch")

    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img
        ehr_feats = input.clinical_indicators
        prompt_text = input.prompt_text
        case_id = input.case_name

        img_token_aligned, ehr_token, img_token_reference = model(
            temporal_img, spatial_img, prompt_text, ehr_feats, idx, case_id
        )

        loss_inter = criterion(img_token_aligned, ehr_token)

        loss_preserve = F.mse_loss(img_token_aligned, img_token_reference)

        loss = loss_inter + 0.05 * loss_preserve
        print("loss inter:", loss_inter)
        print("loss preserve:", loss_preserve)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

    avg_loss = total_loss / total_samples

    return avg_loss

@torch.no_grad()
def validate(model, dataloader, device, cfg, epoch, log_dir, criterion):
    model.eval()
    metrics = {}

    total_loss,  total_samples = 0.0, 0
    dataloader_tqdm = tqdm(dataloader, desc="Validation", unit="batch")

    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img
        ehr_feats = input.clinical_indicators
        prompt_text = input.prompt_text
        case_id = input.case_name

        img_token_aligned, ehr_token, img_token_reference = model(
            temporal_img, spatial_img, prompt_text, ehr_feats, idx, case_id
        )

        loss_inter = criterion(img_token_aligned, ehr_token)

        loss_preserve = F.mse_loss(img_token_aligned, img_token_reference)

        loss = loss_inter + 0.05 * loss_preserve

        total_loss += loss.item() * temporal_img.size(0)
        total_samples += temporal_img.size(0)

    avg_loss = total_loss / total_samples
    save_path = os.path.join(f"{log_dir}", "results", cfg["date"], "projector")

    os.makedirs(save_path, exist_ok=True)

    csv_file = f"{save_path}/projector_result.csv"

    metrics = {
        "epoch": epoch,
        "total_loss": avg_loss,
    }
    fieldnames = ["epoch", "total_loss"]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow(metrics)
        print(f"Metrics saved to {csv_file}")
    return avg_loss

def main_worker(rank, world_size, args):
    try:
        setup_ddp(rank, world_size)

        device = torch.device(f"cuda:{rank}")

        cfg = {
            "date": "0527_attention_viz_prompt_50",
            "model_student": "uniformer_base_IL",
            "num_time_bins": 4,
            "embed_dim": [64, 128, 320, 512],
            "ehr_dim": [13, 7, 12, 5],
            "depth": 24,
            "frames": 24,
            "work_space": work_space,
            "json_file": '../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction.json',
            "datasets": 'survival_time',
            "phase": 'train',
            "crop_size": 96,
            "batch_size": 8,
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
            "use_feature_output": False,
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
        parser.add_argument("--dataset",  type=str, default='CINE')
        args = parser.parse_args()

        set_seed(cfg["seed"])

        save_path = os.path.join(work_space, "results", cfg["date"], "projector")
        os.makedirs(save_path, exist_ok=True)

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_dir = os.path.join(save_path, current_date)
        os.makedirs(log_dir, exist_ok=True)

        cfg["json_file"] = "../Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction_embedded_50_prompt_ft.json"
        cfg["json_file"] = time_bins_and_dataset_split(cfg["json_file"], cfg["num_time_bins"], log_dir, cfg["seed"], type_="ssl_pretrain")

        if cfg["model_student"] == "uniformer_base_IL":
            model = UniformerEncoder(in_chans_temporal=cfg["frames"], variable_stride = cfg["variable_stride"])

            pretrained_backbone_path = f"{work_space}/results/0507_attention_viz/distill/2025-05-14 07:49:45/best_distill_epoch_37.pth"
            if cfg["load_pretrained_weight"]:
                model_backbone = load_backbone_weights(model, pretrained_backbone_path, cfg["load_pretrained_weight"], model_name = "uniformer").to(device)
            else:
                model_backbone = model
            routing_module = PromptMapper()
            trained_routing_module_path = "../shy_semantic_injection_module/checkpoints_9/prompt_mapper_epoch1976.pt"
            routing_module.load_state_dict(torch.load(trained_routing_module_path))

            projector = DINO_Cine_Projector(cfg, model_backbone, routing_module).to(device)
            model = projector

            if cfg["frozen_backbone"]:

                for name, params in model.named_parameters():
                    if 'student' in name:
                        params.requires_grad = False

            for name, params in model.named_parameters():
                if 'routing_module' in name:
                    params.requires_grad = False

            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ Total trainable parameters: {total_trainable_params:,}")

        if cfg["model_student"] == "vqvae":
            model = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans_temporal=cfg["frames"],in_chans_spatial=cfg["depth"], variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05).to(device)
            print(model)

            pretrained_backbone_path = f"{work_space}/results/0416_vqvae/codebook/2025-04-16 15:34:20/best_codebook_epoch_38.pth"

            if cfg["load_pretrained_weight"]:
                model_backbone = load_backbone_weights(model, pretrained_backbone_path, cfg["load_pretrained_weight"], model_name = "vqvae_codebook").to(device)
            else:
                model_backbone = model

            routing_module = PromptMapper()
            trained_routing_module_path = "../shy_semantic_injection_module/checkpoints_9/prompt_mapper_epoch1976.pt"
            routing_module.load_state_dict(torch.load(trained_routing_module_path))

            projector = Cine_Projector(cfg, model_backbone, routing_module).to(device)
            model = projector

            if cfg["frozen_backbone"]:

                for name, params in model.named_parameters():
                    if 'student' in name:
                        params.requires_grad = False
                    if 'routing_module' in name:
                        params.requires_grad = False

            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ Total trainable parameters: {total_trainable_params:,}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        criterion_inter_ehr_img = RelationalDistanceLoss(margin=0.2, distance_metric='cosine').to(device)

        train_dataset = PrognosisDataset(cfg["json_file"], "train", cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"], crop_size=cfg["crop_size"],  transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"])
        val_dataset = PrognosisDataset(cfg["json_file"],  "val", cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"],  crop_size=cfg["crop_size"], transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"])

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=train_sampler, num_workers=16, collate_fn=cine_collate_fn, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], sampler=val_sampler, num_workers=16, collate_fn=cine_collate_fn, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        num_epochs = 100

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_file = os.path.join(log_dir, "config.log")

        logging.getLogger().handlers.clear()
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        logging.info("Configuration: %s", cfg)

        best_loss = 1e8
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion = criterion_inter_ehr_img)
            val_loss = validate(model, val_loader, device, cfg, epoch, log_dir, criterion = criterion_inter_ehr_img)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            result_csv_path = os.path.join(log_dir, "loss_acc.csv")

            if val_loss < best_loss:
                best_loss = val_loss
                for file in glob.glob(os.path.join(log_dir, "best_projector_epoch_*.pth")):
                    os.remove(file)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                }, os.path.join(f"{log_dir}", f"best_projector_epoch_{epoch}.pth"))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
            }, os.path.join(f"{log_dir}", f"best_last_projector.pth"))
        cleanup_ddp()
    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down.")
        return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())

    args = parser.parse_args()

    try:
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Terminating all processes...")

        os.kill(os.getpid(), signal.SIGINT)