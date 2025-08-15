import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import math
from itertools import islice
import collections

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nll_loss(hazards, S, y, c, alpha=0.4, eps=1e-7, reduction='mean'):
    y = y.type(torch.int64).unsqueeze(1)
    c = c.type(torch.int64).unsqueeze(1)

    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

class CrossEntropySurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        super(CrossEntropySurvLoss, self).__init__()
        self.alpha = alpha

    def forward(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

class NLLSurvLoss_dep(nn.Module):
    def __init__(self, alpha=0.15):
        super(NLLSurvLoss_dep, self).__init__()
        self.alpha = alpha

    def forward(self, hazards, S, Y, c, alpha=None):
        if alpha is None:

            return nll_loss(hazards, S, Y, c)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

class CoxSurvLoss(nn.Module):
    def forward(self, hazards, S, c):

        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox

def compute_c_index_with_diff(outputs, label_Y, c, default_time=None):
    valid_idx = (c == 1)
    outputs = outputs[valid_idx]
    label_Y = label_Y[valid_idx]

    num_samples = outputs.size(0)
    if num_samples < 2:
        return 1.0

    differences = outputs[:, 1:] - outputs[:, :-1]

    predicted_labels = torch.argmax(differences, dim=1) + 1

    for i in range(num_samples):
        if torch.all(differences[i] < 0):
            predicted_labels[i] = default_time if default_time is not None else outputs.size(1) - 1

    concordant = 0
    permissible = 0

    for i in range(num_samples):
        for j in range(num_samples):
            if label_Y[i] < label_Y[j]:
                permissible += 1
                if predicted_labels[i] < predicted_labels[j]:
                    concordant += 1
                elif predicted_labels[i] == predicted_labels[j]:
                    concordant += 0.5

    return concordant / permissible if permissible > 0 else 1.0

def compute_c_index_with_risk(all_risks, label_Y, c):
    num_samples = len(label_Y)
    concordant = 0
    permissible = 0

    for i in range(num_samples):
        for j in range(i + 1, num_samples):

            if c[i] == 0 and c[j] == 0:
                continue

            if c[i] == 0 and c[j] == 1:
                if label_Y[i] >= label_Y[j]:
                    permissible += 1
                    risk_i = all_risks[i]
                    risk_j = all_risks[j]
                    if risk_i < risk_j:
                        concordant += 1
                    elif risk_i == risk_j:
                        concordant += 0.5
                continue
            if c[i] == 1 and c[j] == 0:
                if label_Y[j] >= label_Y[i]:
                    permissible += 1
                    risk_i = all_risks[i]
                    risk_j = all_risks[j]
                    if risk_j < risk_i:
                        concordant += 1
                    elif risk_i == risk_j:
                        concordant += 0.5
                continue

            if c[i] == 1 and c[j] == 1:
                permissible += 1
                if label_Y[i] < label_Y[j]:
                    if all_risks[i] > all_risks[j]:
                        concordant += 1
                    elif all_risks[i] == all_risks[j]:
                        concordant += 0.5
                elif label_Y[i] > label_Y[j]:
                    if all_risks[i] < all_risks[j]:
                        concordant += 1
                    elif all_risks[i] == all_risks[j]:
                        concordant += 0.5

    return concordant / permissible if permissible > 0 else 1.0

def compute_c_index_with_threshold(outputs, label_Y, c, threshold=0.8, default_time=None):
    valid_idx = (c == 1)
    outputs = outputs[valid_idx]
    label_Y = label_Y[valid_idx]

    num_samples = outputs.size(0)
    if num_samples < 2:
        return 1.0

    predicted_labels = torch.argmax((outputs >= threshold).int(), dim=1)

    for i in range(num_samples):
        if torch.all(outputs[i] < threshold):
            predicted_labels[i] = default_time if default_time is not None else outputs.size(1) - 1

    concordant = 0
    permissible = 0

    for i in range(num_samples):
        for j in range(num_samples):
            if label_Y[i] < label_Y[j]:
                permissible += 1
                if predicted_labels[i] < predicted_labels[j]:
                    concordant += 1
                elif predicted_labels[i] == predicted_labels[j]:
                    concordant += 0.5

    return concordant / permissible if permissible > 0 else 1.0

if __name__ == "__main__":

    B = 4
    num_time_bins = 5
    outputs = torch.tensor([[0.1, 0.3, 0.2, 0.4, 0.5],
                            [0.2, 0.4, 0.6, 0.8, 1.0],
                            [0.5, 0.4, 0.3, 0.2, 0.1],
                            [0.3, 0.6, 0.9, 0.5, 0.4]])
    label_Y = torch.tensor([0, 2, 1, 3])
    c = torch.tensor([1, 1, 1, 0])

    c_index = compute_c_index_with_threshold(outputs, label_Y, c)
    print(f"C-index: {c_index}")