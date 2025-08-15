
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
from .data_structure import CineModalInput

class PrognosisDataset(Dataset):
    def __init__(self, json_file, phase, in_chans, frames, model, use_crop, crop_size, transform=None, use_augment=True, ones_focused=False, center="renji", apply_spatial_sampling=False):
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
        self.resize_transform = transforms.Resize((crop_size, crop_size))
        self.label_dict = {}
        self.label_4_dict = {}
        self.text_dict = {}
        self.label_time_dict = {}
        self.use_crop = use_crop
        self.use_augment = use_augment
        self.ones_focused = ones_focused
        self.apply_spatial_sampling = apply_spatial_sampling
        self.clinical_indicator_dict = {}
        self.prompt_text_dict = {}

        with open(json_file, 'r') as f:
            data = json.load(f)

        for case in data[phase]:
            if phase != "test":
                for k, v in case.items():
                    if self.ones_focused:
                        if int(v["c"]) == 1:
                            continue

                    if "img" in v.keys():
                        self.case_names.append(k)

                        self.student_img_dict[k] = [v['img'][0], v['img'][0], v['img'][0], v['img'][0]]
                        self.student_optical_img_dict[k] = [item.replace("cropped", "optical_cropped") for item in self.student_img_dict[k]]

                        self.text_dict[k] = ""
                        self.prompt_text_dict[k] = ""
                        self.clinical_indicator_dict[k] = []
                        self.label_dict[k] = [torch.tensor(int(v["c"])), torch.tensor(int(v["label"]))]
                        self.label_time_dict[k] = torch.tensor(float(v["survival_time"]))
            else:
                for k, v in case.items():
                    if self.ones_focused:
                        if int(v["c"]) == 1:
                            continue
                    if v["center"] != center:
                        continue

                    if "img" in v.keys():
                        self.case_names.append(k)

                        self.student_img_dict[k] = [v['img'][0], v['img'][0], v['img'][0], v['img'][0]]
                        self.student_optical_img_dict[k] = [item.replace("cropped", "optical_cropped") for item in self.student_img_dict[k]]

                        self.text_dict[k] = ""
                        self.prompt_text_dict[k] = ""
                        self.clinical_indicator_dict[k] = []
                        self.label_dict[k] = [torch.tensor(int(v["c"])), torch.tensor(int(v["label"]))]
                        self.label_time_dict[k] = torch.tensor(float(v["survival_time"]))

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        student_image_SA, img_2ch, img_3ch, img_4ch = self.student_img_dict[case_name]
        temporal_img, spatial_img = self.load_image(student_image_SA, case_name)
        CH_temporal_img_ch2, CH_temporal_img_ch3, CH_temporal_img_ch4 = self.load_image_CH(img_2ch, img_3ch, img_4ch, case_name)

        student_image_optical_SA, img_optical_2ch, img_optical_3ch, img_optical_4ch = self.student_optical_img_dict[case_name]
        temporal_optical_img, spatial_optical_img = self.load_optical_image(student_image_optical_SA, case_name)
        CH_temporal_img_optical_ch2, CH_temporal_img_optical_ch3, CH_temporal_img_optical_ch4 = self.load_image_CH(img_optical_2ch, img_optical_3ch, img_optical_4ch, case_name)

        label_c, label_Y = self.label_dict[case_name]
        label_time = self.label_time_dict[case_name]

        if self.model == 'DenseNet121Classifier':
            D,T,H,W = student_img.shape
            temporal_img = temporal_img[D//2-1, T//2-1, :,:].unsqueeze(0)

        text = self.text_dict[case_name]
        prompt_text = self.prompt_text_dict[case_name]
        clinical_indicators = self.clinical_indicator_dict[case_name]

        return CineModalInput(
            temporal_img=temporal_img,
            spatial_img=spatial_img,
            img_2ch=CH_temporal_img_ch2,
            img_3ch=CH_temporal_img_ch3,
            img_4ch=CH_temporal_img_ch4,
            temporal_optical_img=temporal_optical_img,
            spatial_optical_img=spatial_optical_img,
            CH_temporal_img_optical_ch2=CH_temporal_img_optical_ch2,
            CH_temporal_img_optical_ch3=CH_temporal_img_optical_ch3,
            CH_temporal_img_optical_ch4=CH_temporal_img_optical_ch4,
            text=text,
            prompt_text=prompt_text,
            clinical_indicators=clinical_indicators,
            labels_c=label_c,
            labels_Y=label_Y,
            labels_time=label_time,
            case_name=case_name,
    )

    def random_rotation(self, img_data):

        if random.random() > 0.7:
            angle = random.choice([90, 180, 270])
            img_data = torch.rot90(img_data, k=angle // 90, dims=[2, 3])
        return img_data

    def random_flip(self, img_data):

        if random.random() > 0.7:
            img_data = torch.flip(img_data, dims=[2])
        if random.random() > 0.7:
            img_data = torch.flip(img_data, dims=[3])
        if random.random() > 0.7:
            img_data = torch.flip(img_data, dims=[1])
        return img_data

    def add_noise(self, img_data):

        if random.random() > 0.7:
            noise = torch.randn_like(img_data) * 0.05
            img_data = img_data + noise
            img_data = torch.clamp(img_data, 0.0, 1.0)
        return img_data

    def resizeCH(self, image, depth, size):

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        image = image.unsqueeze(0).unsqueeze(0)
        image_resized = F.interpolate(image, size=(depth, size, size), mode='trilinear', align_corners=True)
        image_resized = image_resized.squeeze(0).squeeze(0)

        return image_resized

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

    def adjust_contrast(self, img_data, lower=0.8, upper=1.2):

        contrast_factor = torch.empty(1).uniform_(lower, upper).item()
        mean = img_data.mean()
        img_data = (img_data - mean) * contrast_factor + mean
        return img_data

    def random_intensity_shift(self, img_data, max_shift=0.1):

        shift = torch.empty(1).uniform_(-max_shift, max_shift).item()
        img_data = img_data + shift
        img_data = torch.clamp(img_data, 0, 1)
        return img_data

    def apply_augmentation(self, img_data):
        img_data = self.random_flip(img_data)
        img_data = self.random_rotation(img_data)
        img_data = self.add_noise(img_data)
        img_data = self.adjust_contrast(img_data)
        img_data = self.random_intensity_shift(img_data)
        return img_data

    def resize4D(self, img_data):
        if self.apply_spatial_sampling:
            img_data = self.resize3D_spatial_layer(img_data, (self.in_chans, self.crop_size, self.crop_size))
        else:
            img_data = self.resize3D(img_data, (self.in_chans, self.crop_size, self.crop_size))
        img_data = torch.tensor(img_data, dtype=torch.float32)
        img_data = self.resize_temporal(img_data, self.frames, self.in_chans, self.crop_size)

        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
        img_data = self.resize_transform(img_data)

        if self.model != 'cascade_uniformer' and self.model != 'RoPECascadeUniformer' and self.model != 'TumorUniformer':
            img_data = img_data.permute(1, 0, 2, 3)

        return img_data

    def load_image(self, img_file, case_name):
        if self.use_crop:
            img_file = img_file.replace('0.nii.gz', 'cropped.nii.gz')

        img_data = nib.load(f"{img_file}").get_fdata()
        if self.model == 'TumorUniformer':
            H,W,D = img_data.shape
            img_data = np.expand_dims(img_data, axis=2)

        img_data = img_data.transpose(2, 3, 1, 0)
        temporal_img = img_data
        temporal_img = self.resize4D(temporal_img)

        spatial_img = img_data.transpose(1, 0, 2, 3)
        spatial_img = self.resize4D(spatial_img)

        if self.phase == "train" and self.use_augment:
            temporal_img = self.apply_augmentation(temporal_img)
            spatial_img = self.apply_augmentation(spatial_img)
        return temporal_img, spatial_img

    def load_image_CH(self, img_2ch, img_3ch, img_4ch, case_name):
        if self.use_crop:
            img_2ch, img_3ch, img_4ch = img_2ch.replace('0.nii.gz', 'cropped.nii.gz'), img_3ch.replace('0.nii.gz', 'cropped.nii.gz'), img_4ch.replace('0.nii.gz', 'cropped.nii.gz')

        img_data_2ch, img_data_3ch, img_data_4ch = nib.load(f"{img_2ch}").get_fdata(), nib.load(f"{img_3ch}").get_fdata(), nib.load(f"{img_4ch}").get_fdata()

        CH_temporal_img_list = []
        for img_data in [img_data_2ch, img_data_3ch, img_data_4ch]:
            img_data = img_data.transpose(2, 3, 1, 0)

            img_data = torch.tensor(img_data, dtype=torch.float32)
            img_data = torch.mean(img_data, dim=1)
            img_data = self.resizeCH(img_data, self.frames, self.crop_size)
            img_data = img_data.squeeze()

            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
            img_data = self.resize_transform(img_data)

            CH_temporal_img_list.append(img_data.unsqueeze(0))
        return CH_temporal_img_list[0], CH_temporal_img_list[1], CH_temporal_img_list[2]

    def load_optical_image(self, img_file, case_name):

        img_data = nib.load(f"{img_file}").get_fdata()

        if self.model == 'TumorUniformer':
            H,W,D = img_data.shape
            img_data = np.expand_dims(img_data, axis=2)

        img_data = img_data.transpose(2, 3, 1, 0)

        temporal_img = img_data
        temporal_img = self.resize4D(temporal_img)

        spatial_img = img_data.transpose(1, 0, 2, 3)
        spatial_img = self.resize4D(spatial_img)

        return temporal_img, spatial_img

    def load_optical_image_CH(self, img_2ch, img_3ch, img_4ch, case_name):
        img_data_2ch, img_data_3ch, img_data_4ch = nib.load(f"{img_2ch}").get_fdata(), nib.load(f"{img_3ch}").get_fdata(), nib.load(f"{img_4ch}").get_fdata()

        CH_temporal_img_list = []
        for img_data in [img_data_2ch, img_data_3ch, img_data_4ch]:
            img_data = img_data.transpose(2, 3, 1, 0)

            img_data = torch.tensor(img_data, dtype=torch.float32)
            img_data = torch.mean(img_data, dim=1)
            img_data = self.resizeCH(img_data, self.frames, self.crop_size)
            img_data = img_data.squeeze()

            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
            img_data = self.resize_transform(img_data)

            CH_temporal_img_list.append(img_data.unsqueeze(0))

        return CH_temporal_img_list[0], CH_temporal_img_list[1], CH_temporal_img_list[2]