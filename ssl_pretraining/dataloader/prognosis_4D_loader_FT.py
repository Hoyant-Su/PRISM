
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
    def __init__(self, json_file, phase, in_chans, frames, model, use_crop, crop_size, transform=None, use_augment=True, ones_focused=False, center="renji"):
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
                        print(k)
                        self.prompt_text_dict[k] = v['prompt_text']
                        self.student_img_dict[k] = v['img'][0]
                        self.clinical_indicator_dict[k] = v["clinical_data"]
                        self.label_dict[k] = [torch.tensor(int(v["c"])), torch.tensor(int(v["label"]))]
                        self.label_time_dict[k] = torch.tensor(float(v["survival_time"]))
                    else:
                        continue
                else:
                    continue

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        student_image_SA = self.student_img_dict[case_name]
        temporal_img, spatial_img = self.load_image(student_image_SA, case_name)

        prompt_text = self.prompt_text_dict[case_name]
        label_c, label_Y = self.label_dict[case_name]
        label_time = self.label_time_dict[case_name]
        

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
            clinical_indicators=clinical_indicators,
            labels_c=label_c,
            labels_Y=label_Y,
            labels_time=label_time,
            case_name=case_name,
            prompt_text=prompt_text,
    )

    def random_rotation(self, img_data):
        # Random rotation in the (H, W) plane
        if random.random() > 0.7:
            angle = random.choice([90, 180, 270])
            img_data = torch.rot90(img_data, k=angle // 90, dims=[2, 3])
        return img_data

    def random_flip(self, img_data):
        # Random flip along spatial axes
        if random.random() > 0.7:
            img_data = torch.flip(img_data, dims=[2])
        if random.random() > 0.7:
            img_data = torch.flip(img_data, dims=[3])
        if random.random() > 0.7:
            img_data = torch.flip(img_data, dims=[1])
        return img_data

    def add_noise(self, img_data):
        # Add random Gaussian noise
        if random.random() > 0.7:
            noise = torch.randn_like(img_data) * 0.05
            img_data = img_data + noise
            img_data = torch.clamp(img_data, 0.0, 1.0)  # Ensure values remain in [0, 1]
        return img_data
        
      
    def resize3D(self, image, size):
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (T, D, H, W)
        T, D, H, W = image.shape
        image_resized = torch.zeros((T, size[0], size[1], size[2]), dtype=torch.float32)  # (T', D', H', W')
        
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D, H, W)
        
        for t in range(image.shape[2]):
            frame = image[:, :, t, :, :, :]
            frame_resized = F.interpolate(frame, size=(size[0], size[1], size[2]), mode='trilinear', align_corners=True)
            image_resized[t, :, :, :] = frame_resized
        
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

    def adjust_contrast(self, img_data, lower=0.8, upper=1.2):
        contrast_factor = torch.empty(1).uniform_(lower, upper).item()
        mean = img_data.mean()
        img_data = (img_data - mean) * contrast_factor + mean
        return img_data

    def random_intensity_shift(self, img_data, max_shift=0.1):
        shift = torch.empty(1).uniform_(-max_shift, max_shift).item()
        img_data = img_data + shift
        img_data = torch.clamp(img_data, 0, 1)  # Ensure values remain in [0, 1]
        return img_data


    def apply_augmentation(self, img_data):
        img_data = self.random_flip(img_data)
        img_data = self.random_rotation(img_data)
        img_data = self.add_noise(img_data)
        img_data = self.adjust_contrast(img_data)
        img_data = self.random_intensity_shift(img_data)
        return img_data


    def resize4D(self, img_data):
        img_data = self.resize3D(img_data, (self.in_chans, self.crop_size, self.crop_size))
        img_data = torch.tensor(img_data, dtype=torch.float32)
        img_data = self.resize_temporal(img_data, self.frames, self.in_chans, self.crop_size)
        # Normalize to [0, 1]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
        img_data = self.resize_transform(img_data)
        
        if self.model != 'cascade_uniformer' and self.model != 'RoPECascadeUniformer' and self.model != 'TumorUniformer':
            img_data = img_data.permute(1, 0, 2, 3)  # (T, D, H, W)-> (D, T, H, W)
        
        return img_data


    def load_image(self, img_file, case_name):
        if self.use_crop:
            img_file = img_file.replace('0.nii.gz', 'cropped.nii.gz')
        
        #img_file = img_file.replace('CINE-SA', 'PSIR')
        
        img_data = nib.load(f"{img_file}").get_fdata() #(H, W, T, D)

        if self.model == 'TumorUniformer':
            H,W,D = img_data.shape
            img_data = np.expand_dims(img_data, axis=2)

        img_data = img_data.transpose(2, 3, 1, 0)  # (H, W, T, D) -> (T, D, H, W)
        
        temporal_img = img_data
        temporal_img = self.resize4D(temporal_img)

        spatial_img = img_data.transpose(1, 0, 2, 3)
        spatial_img = self.resize4D(spatial_img)

        if self.phase == "train" and self.use_augment:
            temporal_img = self.apply_augmentation(temporal_img)
            spatial_img = self.apply_augmentation(spatial_img)
        return temporal_img, spatial_img

