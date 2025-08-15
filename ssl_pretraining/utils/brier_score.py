import torch
import numpy as np

def calc_brier_score(survival, c, Y, B, N):
    brier_scores = []
    for i in range(B):
        survival_i = survival[i]
        y_i = np.ones(N)
        if c[i] == 0:
            y_i[Y[i]:] = 0

        valid_time_periods = slice(0, Y[i] + 1)
        num_valid_periods = Y[i] + 1

        bs = np.sum((survival_i[valid_time_periods] - y_i[valid_time_periods]) ** 2) / num_valid_periods
        brier_scores.append(bs)

    return np.mean(np.array(brier_scores))

if __name__ == "__main__":

    B = 3
    N = 4
    survival = np.array([[1, 1, 0, 0],
                        [1, 1, 1, 1],
                        [0.95, 0, 0, 0]])
    c = np.array([0, 1, 0])
    Y = np.array([2, 0, 1])

    brier_scores = calc_brier_score(survival, c, Y, B, N)
    print(f"Brier Scores for each sample: {brier_scores}")
