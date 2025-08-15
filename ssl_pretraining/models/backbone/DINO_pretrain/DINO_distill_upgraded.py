import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import sys
work_space = 'ssl_pretraining'
sys.path.append(work_space)

from models.backbone.compare_models.uniformer import uniformer_base_IL
from models.backbone.compare_models.uniformer_optical_included import OpticalPromptedUniFormer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(student_logits, teacher_logits, epoch):
    logits_combined = torch.cat([student_logits, teacher_logits], dim=0).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    logits_2d = tsne.fit_transform(logits_combined)

    x_min, x_max = logits_2d[:, 0].min(), logits_2d[:, 0].max()
    y_min, y_max = logits_2d[:, 1].min(), logits_2d[:, 1].max()

    logits_2d[:, 0] = 2 * (logits_2d[:, 0] - x_min) / (x_max - x_min) - 1
    logits_2d[:, 1] = 2 * (logits_2d[:, 1] - y_min) / (y_max - y_min) - 1

    batch_size = student_logits.size(0)
    colors = np.repeat(np.arange(batch_size), 2)

    plt.figure(figsize=(5, 4))
    plt.scatter(logits_2d[:, 0], logits_2d[:, 1], c=colors, cmap='tab10', alpha=0.7)

    plt.title(f't-SNE of Student and Teacher logits (Epoch {epoch})', fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    cbar = plt.colorbar()
    cbar.set_label('Batch Sample ID')

    plt.savefig(f'tsne_epoch_{epoch}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

class DINODistill(nn.Module):
    def __init__(self, cfg):
        super(DINODistill, self).__init__()
        self.student, self.teacher = self.build_model(cfg, 'student'), self.build_model(cfg, 'teacher')

        self.teacher.eval()

    def build_model(self, cfg, model_type):
        model_arch = cfg["model"]

        if cfg["model"] == "uniformer_base_IL":
            if model_type == 'student':
                model = uniformer_base_IL(
                    in_chans = cfg["in_chans_student"],

                    num_classes = cfg["num_time_bins"],
                    variable_stride = cfg["variable_stride"]

                )
            elif model_type == 'teacher':
                model = uniformer_base_IL(

                    in_chans = 1,
                    num_classes = cfg["num_time_bins"],
                    variable_stride = cfg["variable_stride"]

                )
        elif cfg["model"] == "OpticalPromptedUniFormer":
            if model_type == "student":
                model = OpticalPromptedUniFormer(num_classes = cfg["num_time_bins"],variable_stride = cfg["variable_stride"], in_chans=cfg["in_chans_student"])
            elif model_type == "teacher":
                model = OpticalPromptedUniFormer(num_classes = cfg["num_time_bins"],variable_stride = cfg["variable_stride"], in_chans=1)

        return model

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, data_SA, data_ch2, data_ch3, data_ch4,
                opt_data_SA, opt_data_ch2, opt_data_ch3, opt_data_ch4,
                teacher_temp=0.07, contrast_temp=0.1, step=None, epoch=None):

        student_output = self.student(data_SA, opt_data_SA)

        teacher_output_list = []

        for (data, opt_data) in [(data_ch2, opt_data_ch2), (data_ch3, opt_data_ch3), (data_ch4, opt_data_ch4)]:
            teacher_output_list.append(self.teacher(data, opt_data))
        teacher_output = torch.stack(teacher_output_list).mean(dim=0)

        student_logits, teacher_logits = student_output / teacher_temp, teacher_output / teacher_temp

        student_log_probs = F.log_softmax(student_logits, dim=-1)

        distillation_loss = -torch.mean(torch.sum(student_log_probs * F.softmax(teacher_logits, dim=-1), dim=-1))

        student_norm = F.normalize(student_output, p=2, dim=-1)

        student_similarity = torch.matmul(student_norm, student_norm.t())
        labels = torch.arange(student_similarity.size(0)).to(student_similarity.device)
        contrastive_student_loss = F.cross_entropy(student_similarity / contrast_temp, labels)

        total_loss = distillation_loss + contrastive_student_loss

        if step == 0:
            plot_tsne(student_logits, teacher_logits, epoch)

        return total_loss

    def update_teacher(self, momentum):

        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
