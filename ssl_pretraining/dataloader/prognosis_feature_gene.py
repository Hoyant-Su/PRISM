
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from .data_structure_ft import CineModalInput

class PrognosisDataset(Dataset):
    def __init__(self, json_file, phase, in_chans, frames, model, use_crop, crop_size, transform=None, use_augment=True, ones_focused=False, apply_spatial_sampling = None, prompt_enable=True):
        self.transform = transform
        self.in_chans = in_chans
        self.crop_size = crop_size
        self.frames = frames
        self.model = model
        self.phase = phase
        self.teacher_img_dict = {}
        self.student_img_dict = {}

        self.student_optical_img_dict = {}
        self.case_names = []
        self.prompt_text_dict = {}
        self.resize_transform = transforms.Resize((crop_size, crop_size))
        self.label_dict = {}
        self.label_4_dict = {}
        self.text_dict = {}
        self.label_time_dict = {}
        self.use_crop = use_crop
        self.use_augment = use_augment
        self.ones_focused = ones_focused
        self.clinical_indicator_dict = {}

        self.apply_spatial_sampling = apply_spatial_sampling

        with open(json_file, 'r') as f:
            data = json.load(f)

        for case in data[phase]:
            for k, v in case.items():
                if self.ones_focused:
                    if int(v["c"]) == 1:
                        continue

                if "img" in v.keys() and "clinical_data" in v.keys():
                    if 'CINE-SA' in v["img"][0] and len(v["clinical_data"]) != 0:
                        self.case_names.append(k)
                        self.student_img_dict[k] = v['img'][0]
                        self.clinical_indicator_dict[k] = v["clinical_data"]
                        self.student_optical_img_dict[k] = self.student_img_dict[k].replace("cropped", "optical_cropped")
                        self.text_dict[k] = "pass"
                        if prompt_enable:
                            print(v['prompt_text'])
                            self.prompt_text_dict[k] = v['prompt_text']
                        else:
                            self.prompt_text_dict[k] = "pass"

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        prompt_text = self.prompt_text_dict[case_name]
        student_image_SA = self.student_img_dict[case_name]
        temporal_img, spatial_img = self.load_image(student_image_SA, case_name)
        student_image_optical_SA = self.student_optical_img_dict[case_name]
        temporal_optical_img, spatial_optical_img = self.load_optical_image(student_image_optical_SA, case_name)
        text = self.text_dict[case_name]
        prompt_text = self.prompt_text_dict[case_name]

        if self.model == 'DenseNet121Classifier':
            D,T,H,W = student_img.shape
            temporal_img = temporal_img[D//2-1, T//2-1, :,:].unsqueeze(0)

        clinical_indicators = self.clinical_indicator_dict[case_name]
        clinical_indicators = torch.tensor(
            [float(x) for x in clinical_indicators],
            dtype=torch.float
        ).unsqueeze(0)

        return CineModalInput(
            temporal_img=temporal_img,
            spatial_img=spatial_img,
            temporal_optical_img=temporal_optical_img,
            spatial_optical_img=spatial_optical_img,
            text=text,
            prompt_text=prompt_text,
            clinical_indicators=clinical_indicators,
            case_name=case_name,
    )

    def resize3D(self, image, size):
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        T, D, H, W = image.shape
        image_resized = torch.zeros((T, size[0], size[1], size[2]), dtype=torch.float32)

        image = image.unsqueeze(0).unsqueeze(0)

        for t in range(image.shape[2]):
            frame = image[:, :, t, :, :, :]
            frame_resized = F.interpolate(frame, size=(size[0], size[1], size[2]), mode='trilinear', align_corners=True)
            image_resized[t, :, :, :] = frame_resized

        return image_resized.cpu().numpy()

    def resize3D_spatial_layer(self, image, size):
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        T, D, H, W = image.shape

        layers_selected = image[:, [0, D // 2, D - 1], :, :]

        image_resized = torch.zeros((T, 3, size[1], size[2]), dtype=torch.float32)

        for t in range(T):
            frame = layers_selected[t, :, :, :]
            frame_resized = F.interpolate(frame.unsqueeze(0), size=(size[1], size[2]), mode='bilinear', align_corners=True)
            image_resized[t, :, :, :] = frame_resized.squeeze(0)

        return image_resized.cpu().numpy()

    def resize_temporal(self, tensor, target_T, target_D, crop_size):
        T, D, H, W = tensor.shape

        if T > target_T:
            indices = torch.linspace(0, T-1, target_T).long()
            tensor = tensor[indices, :, :, :]
        elif T < target_T:
            pad_T = target_T - T
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, pad_T), mode='constant')

        return tensor

    def resize4D(self, img_data):
        if self.apply_spatial_sampling:
            img_data = self.resize3D_spatial_layer(img_data, (self.in_chans, self.crop_size, self.crop_size))
        else:
            img_data = self.resize3D(img_data, (self.in_chans, self.crop_size, self.crop_size))
        img_data = torch.tensor(img_data, dtype=torch.float32)
        img_data = self.resize_temporal(img_data, self.frames, self.in_chans, self.crop_size)

        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
        img_data = self.resize_transform(img_data)

        if self.model != 'cascade_uniformer' and self.model != 'RoPECascadeUniformer':
            img_data = img_data.permute(1, 0, 2, 3)

        return img_data

    def load_image(self, img_file, case_name):
        if self.use_crop:
            img_file = img_file.replace('0.nii.gz', 'cropped.nii.gz')

        img_data = nib.load(f"{img_file}").get_fdata()
        img_data = img_data.transpose(2, 3, 1, 0)

        temporal_img = img_data
        temporal_img = self.resize4D(temporal_img)

        spatial_img = img_data.transpose(1, 0, 2, 3)
        spatial_img = self.resize4D(spatial_img)

        return temporal_img, spatial_img

    def load_optical_image(self, img_file, case_name):

        img_data = nib.load(f"{img_file}").get_fdata()
        img_data = img_data.transpose(2, 3, 1, 0)

        temporal_img = img_data
        temporal_img = self.resize4D(temporal_img)

        spatial_img = img_data.transpose(1, 0, 2, 3)
        spatial_img = self.resize4D(spatial_img)
        return temporal_img, spatial_img