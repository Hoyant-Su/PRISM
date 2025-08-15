import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import sys
work_space = 'ssl_pretraining'
sys.path.append(work_space)

from models.backbone.compare_models.uniformer import uniformer_base_IL
from models.backbone.visual_encoder import UniformerEncoder
from utils.backbone_weight_loader import load_backbone_weights

class DINODistill(nn.Module):
    def __init__(self, cfg):
        super(DINODistill, self).__init__()
        self.student, self.teacher = self.build_model(cfg, 'student'), self.build_model(cfg, 'teacher')
        self.teacher.eval()

        self.center_momentum = cfg.get("center_momentum", 0.79)
        self.teacher_temp = cfg.get("teacher_temp", 0.05)
        self.register_buffer('center', torch.zeros(1, 512))

    def build_model(self, cfg, model_type):
        if model_type == 'student' and cfg["model"] == "uniformer_base_IL":
            model = UniformerEncoder(in_chans_temporal=cfg["in_chans_student"], variable_stride = cfg["variable_stride"])
            if cfg["load_pretrained_weight"]:
                pretrained_backbone_path = "ssl_pretraining/weights/uniformer_base_k400_8x8_partial.pth"
                model = load_backbone_weights(model, pretrained_backbone_path, cfg["load_pretrained_weight"])

        elif model_type == 'teacher' and cfg["model"] == "uniformer_base_IL":

            model = UniformerEncoder(in_chans_temporal=1, variable_stride = cfg["variable_stride"])

        if model_type == 'teacher':
            for param in model.parameters():
                param.requires_grad = False
        return model

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, data_SA, data_ch2, data_ch3, data_ch4, student_temp=None, step=None, epoch=None):
        student_output = self.student(data_SA)[-1].flatten(2).mean(-1)
        with torch.no_grad():
            teacher_output_list = []
            for data_idx, data in enumerate([data_ch2, data_ch3, data_ch4]):
                teacher_output_list.append(self.teacher(data)[-1].flatten(2).mean(-1))
            teacher_output_raw = torch.stack(teacher_output_list).mean(dim=0)
            self.update_center(teacher_output_raw)
            teacher_output_centered = teacher_output_raw - self.center

        if student_temp is None:
            student_temp = self.teacher_temp

        student_logits = student_output / student_temp
        teacher_logits = teacher_output_centered / self.teacher_temp
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        loss = -torch.mean(torch.sum(teacher_log_probs * student_probs, dim=-1))
        return loss

    def update_teacher(self, momentum):
        with torch.no_grad():

            student_params = self.student.module.parameters() if isinstance(self.student, (DataParallel, nn.parallel.DistributedDataParallel)) else self.student.parameters()
            teacher_params = self.teacher.module.parameters() if isinstance(self.teacher, (DataParallel, nn.parallel.DistributedDataParallel)) else self.teacher.parameters()

            for student_param, teacher_param in zip(student_params, teacher_params):
                teacher_param.data.mul_(momentum).add_((1 - momentum) * student_param.data)