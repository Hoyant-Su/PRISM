import torch
import torch.nn as nn
import torch.nn.functional as F

class TruncationModule(nn.Module):
    def __init__(self, num_time_bins):
        super(TruncationModule, self).__init__()
        self.num_time_bins = num_time_bins

        self.truncation_predictor = nn.Linear(num_time_bins, num_time_bins)

    def forward(self, hazard_values, labels_Y):
        truncation_pos = self.truncation_predictor(hazard_values)
        truncation_pos = torch.sigmoid(truncation_pos)
        truncation_pos = torch.argmax(truncation_pos, dim=1, keepdim=True)

        S = torch.cumprod(1 - hazard_values, dim=1)

        for b in range(S.size(0)):
            trunc_point = int(truncation_pos[b].item())
            if trunc_point < self.num_time_bins:
                S[b, trunc_point:] = S[b, trunc_point:] * torch.exp(-torch.arange(0, self.num_time_bins - trunc_point).float().to(S.device))

        truncation_loss = self.compute_loss(truncation_pos, labels_Y)
        return truncation_pos, S, truncation_loss

    def compute_loss(self, truncation_pos, labels_Y):

        truncation_loss = F.mse_loss(truncation_pos.squeeze().float(), labels_Y.float())
        print(truncation_pos)
        print(labels_Y)
        return truncation_loss

if __name__ == "__main__":
    model = TruncationModule(num_time_bins=4, labels_Y=torch.randn([1,2,3,4,5]))
    hazard_values = torch.rand([5,4])
    output_pos, output_S, loss = model(hazard_values)
    print(output_pos)
    print(output_S)
    print(loss)