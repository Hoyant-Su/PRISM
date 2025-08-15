import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import sys
work_space = 'ssl_pretraining'
sys.path.append(work_space)

from models.backbone.compare_models.uniformer import uniformer_base_IL

class DINODistill(nn.Module):
    def __init__(self, cfg):
        super(DINODistill, self).__init__()
        self.student, self.teacher = self.build_model(cfg, 'student'), self.build_model(cfg, 'teacher')

        self.teacher.eval()

    def build_model(self, cfg, model_type):
        model_arch = cfg["model"]

        if model_type == 'student' and cfg["model"] == "uniformer_base_IL":
            model = uniformer_base_IL(
                in_chans = cfg["in_chans_student"],

                num_classes = cfg["num_time_bins"],
                variable_stride = cfg["variable_stride"]

            )
        elif model_type == 'teacher' and cfg["model"] == "uniformer_base_IL":
            model = uniformer_base_IL(

                in_chans = 1,
                num_classes = cfg["num_time_bins"],
                variable_stride = cfg["variable_stride"]

            )
        return model

    def forward(self, data_SA, data_ch2, data_ch3, data_ch4, teacher_temp=0.07, step=None, epoch=None):
        student_output = self.student(data_SA)
        teacher_output_list = []
        for data in [data_ch2, data_ch3, data_ch4]:
            teacher_output_list.append(self.teacher(data))
        teacher_output = torch.stack(teacher_output_list).mean(dim=0)
        student_logits, teacher_logits = student_output / teacher_temp, teacher_output/ teacher_temp
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        loss = -torch.mean(torch.sum(teacher_log_probs * F.softmax(student_logits, dim=-1), dim=-1))

        return loss

    def update_teacher(self, momentum):
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
