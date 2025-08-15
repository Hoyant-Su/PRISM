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
import pandas as pd
work_space = '../ssl_pretraining'

sys.path.append(work_space)
sys.path.append(f"{work_space}/dataloader")
from models.backbone.Recon_vqvae import VQ_VAE_4D
from models.head.survival_prediction import Cine_Survival_Prediction_Head

from models.head.projector import DINO_Cine_Projector
from models.head.routing_module import PromptMapper
from models.backbone.text_encoder import Rad_TextEncoder
from models.backbone.compare_models.uniformer_optical_included import OpticalPromptedUniFormer
from models.backbone.compare_models.uniformer import uniformer_base_IL
from models.backbone.compare_models.Med3D import resnet200
from models.backbone.compare_models.pcrl_model_3d import PCRLModel3d
from models.backbone.visual_encoder import UniformerEncoder
from dataloader.data_structure import CineModalInput, cine_collate_fn
from dataloader.prognosis_feature_gene import PrognosisDataset
from dataloader.equal_time_bins_split import time_bins_and_dataset_split

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import argparse

from utils.survival_loss import NLLSurvLoss_dep, CrossEntropySurvLoss, compute_c_index_with_threshold, compute_c_index_with_diff, compute_c_index_with_risk
from sksurv.metrics import concordance_index_censored
from utils.metric_visual import plot_kaplan_meier
from utils.brier_score import calc_brier_score
from utils.pesudo_learning_module import TruncationModule
from utils.backbone_weight_loader import load_backbone_weights
from utils.prompt_inference_json_clean import prompt_injection
import time

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

@torch.no_grad()
def feature_gene(model, dataloader, device, cfg, save_path, csv_tail):
    model.eval()

    num_time_bins = cfg["num_time_bins"]
    batch_size = cfg["batch_size"]
    model_name = cfg["model_student"]

    dataloader_tqdm = tqdm(dataloader, desc="Feature_Gene", unit="batch")
    data_to_save = []
    for idx, input in enumerate(dataloader_tqdm):
        input = input.to(device)
        temporal_img = input.temporal_img
        spatial_img = input.spatial_img
        ehr_feats = input.clinical_indicators
        prompt_text = input.prompt_text
        text = input.text
        temporal_opt_img = input.temporal_optical_img
        case_id = input.case_name

        if model_name == 'vqvae':

            outputs, _ = model(temporal_img, spatial_img)
        elif model_name == 'RadBert':
            outputs = model(text)
        elif model_name == "OpticalPromptedUniFormer":
            outputs = model(temporal_img, temporal_opt_img)
        elif model_name == "PCRL":
            B, D, T, H, W = temporal_img.shape
            outputs = model(temporal_img[:,D//2:D//2+1,:,:,:])[0]
        elif model_name == "vqvae_projector":
            outputs, _, _ = model(temporal_img, spatial_img, prompt_text, ehr_feats)
        elif model_name == "uniformer_projector":
            outputs, _, _ = model(temporal_img, spatial_img, prompt_text, ehr_feats, idx, case_id)

        else:
            outputs = model(temporal_img)

        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        for i in range(outputs.shape[0]):
            feature = outputs[i].cpu().numpy()
            case_id_value = case_id[i]

            row = {'img_name': case_id_value}

            for j in range(outputs.shape[1]):
                row[f'feat_{j+1}'] = feature[j]

            data_to_save.append(row)

    df = pd.DataFrame(data_to_save)
    df.to_csv(f'{save_path}/output_features_{csv_tail}.csv', index=False)
    print("CSV file with features has been saved!")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "date": "0321_gulou_feat",
        "model_student": "uniformer_projector",
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
        "batch_size": 16,
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
        "seed": 37,
        "use_distill_model": True,
        "use_pesudo_label": True,
        "use_feature_output": True,
        "apply_spatial_sampling": False,
        "prompt_inference_mode": True,
    } 

    trained_model_path = "" #projector (after stage II) checkpoint path

    log_dir = os.path.dirname(trained_model_path)
    os.makedirs(log_dir, exist_ok=True)

    cfg["json_file"] = '/Monai/Data_process_dataset/full_data_json_prepare/total_survival_prediction_embedded.json'

    cfg["json_file"] = time_bins_and_dataset_split(cfg["json_file"], cfg["num_time_bins"], log_dir, cfg["seed"], type_ = "feature")

    if cfg["prompt_inference_mode"] == True:

        injecting_prompt_list = [
                                 "Concentrate on <Pharmaceutical> and <Physiological> alongside <clinical> and <biochemical> for enhanced examination.",

                                 ]

        cfg["json_file"] = prompt_injection(cfg["json_file"], injecting_prompt_list)

    if isinstance(cfg["json_file"], str):
        cfg["json_file"] = [cfg["json_file"]]

    for json_file in cfg["json_file"]:

        if cfg["apply_spatial_sampling"]:
            cfg["depth"] = 3

        if cfg["model_student"] == "uniformer_base_IL":

            model = UniformerEncoder(in_chans_temporal=cfg["depth"], variable_stride = cfg["variable_stride"], use_feature_output = cfg["use_feature_output"])

            if cfg["use_distill_model"]:

                model = load_backbone_weights(model, trained_model_path, True, model_name = "uniformer").to(device)

        elif cfg["model_student"] == "OpticalPromptedUniFormer":
            model = OpticalPromptedUniFormer(num_classes = cfg["num_time_bins"], variable_stride = cfg["variable_stride"], in_chans=cfg["depth"], use_feature_output=True)

            if cfg["use_distill_model"]:
                model = load_backbone_weights(model, trained_model_path, True, model_name = "OpticalPromptedUniFormer").to(device)

        elif cfg["model_student"] == "vqvae":
            model = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans_temporal=cfg["depth"], in_chans_spatial=cfg["frames"], variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05).to(device)
            survival_prediction_head = Cine_Survival_Prediction_Head(cfg, model).to(device)
            model = survival_prediction_head

            model = load_backbone_weights(model, trained_model_path, True, model_name = "vqvae").to(device)

        elif cfg["model_student"] == "vqvae_projector":
            model_backbone = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans_temporal=cfg["frames"],in_chans_spatial=cfg["depth"], variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05).to(device)
            routing_module = PromptMapper()
            projector = Cine_Projector(cfg, model_backbone, routing_module).to(device)
            model = projector
            model = load_backbone_weights(model, trained_model_path, True, model_name = "vqvae_projector")

        elif cfg["model_student"] == "uniformer_projector":
            model_backbone = UniformerEncoder(in_chans_temporal=cfg["frames"], variable_stride = cfg["variable_stride"])
            routing_module = PromptMapper()
            projector = DINO_Cine_Projector(cfg, model_backbone, routing_module, mode="inference", log_dir=log_dir).to(device)
            model = projector
            model = load_backbone_weights(model, trained_model_path, True, model_name = "uniformer_projector")

        elif cfg["model_student"] == "Med3D":
            model = resnet200(num_classes=cfg["num_time_bins"]).to(device)

        elif cfg["model_student"] == 'RadBert':
            model = Rad_TextEncoder(num_classes = cfg["num_time_bins"], frozen_text_encoder=False, type_ = "feature").to(device)

        elif cfg["model_student"] == "PCRL":
            model = PCRLModel3d()
            model = load_backbone_weights(model, trained_model_path, True, model_name="PCRL").to(device)

        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'Total params: {total_params}; Trainable params: {trainable_params}')

        prompt_enable = True if "projector" in cfg["model_student"] else False
        feature_gene_dataset = PrognosisDataset(json_file,  "train", cfg["depth"], cfg["frames"], cfg["model_student"], cfg["use_crop"],  crop_size=cfg["crop_size"], transform=None, use_augment=cfg["use_augment"], ones_focused = cfg["ones_focus"], apply_spatial_sampling = cfg["apply_spatial_sampling"], prompt_enable=prompt_enable)
        feature_gene_loader = DataLoader(feature_gene_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=16, collate_fn=cine_collate_fn)

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

        csv_tail = json_file.split('_')[-1].strip('.json')
        feature_gene(model, feature_gene_loader, device, cfg, log_dir, csv_tail)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("total time used:", - start_time + end_time)