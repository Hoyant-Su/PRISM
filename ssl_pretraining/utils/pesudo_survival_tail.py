import torch
import numpy as np
from scipy.optimize import curve_fit

def bounded_exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def generate_pseudo_labels(S, labels_Y):
    S_np = S.cpu().numpy()
    labels_Y_np = labels_Y.cpu().numpy()

    batch_size, n = S_np.shape
    S_filled = S_np.copy()

    for i in range(batch_size):
        y = int(labels_Y_np[i])
        if y >= n - 1:
            continue

        x_valid = np.arange(y + 1)
        s_valid = S_filled[i, :y + 1]

        try:
            params, _ = curve_fit(bounded_exponential_decay, x_valid, s_valid, p0=[1.0, 0.1], bounds=(0, [np.inf, np.inf]))

            x_pred = np.arange(y + 1, n)
            s_pred = bounded_exponential_decay(x_pred, *params)
            S_filled[i, y + 1:] = s_pred
        except RuntimeError:
            s_pred = np.linspace(s_valid[-1], 0, n - y - 1)
            S_filled[i, y + 1:] = s_pred

    S_filled = torch.tensor(S_filled, dtype=S.dtype, device=S.device)
    return S_filled

def pesudo_sigmoid(S, device):
    B, N = S.shape

    indices = torch.arange(N).float()
    pseudo_labels = torch.sigmoid(indices).to(device)
    pseudo_labels = pseudo_labels.unsqueeze(0).expand(B, -1)
    S_weighted = S * pseudo_labels
    return S_weighted

def generate_pseudo_labels_with_decay(S, hazard, threshold=0.5):
    hazard = hazard.cpu().numpy().copy()
    S_np = S.cpu().numpy()
    batch_size, n = S_np.shape
    S_filled = S_np.copy()

    for i in range(batch_size):

        diffs = np.diff(hazard[i, :])

        for j in range(1, len(diffs)):
            if diffs[j-1] > threshold:

                x_pred = np.arange(j, n)
                a = S_filled[i, j]
                b = 0.1

                s_pred = bounded_exponential_decay(x_pred - j, a, b)
                S_filled[i, j:] = s_pred

    S_filled = torch.tensor(S_filled, dtype=S.dtype, device=S.device)
    return S_filled

if __name__ == "__main__":

    batch_size = 4
    n = 10
    S = torch.rand(batch_size, n).cuda()
    labels_Y = torch.tensor([3, 5, 7, 2]).cuda()

    S_filled = generate_pseudo_labels(S, labels_Y)

    print("Original S:")
    print(S)
    print("\nFilled S:")
    print(S_filled)