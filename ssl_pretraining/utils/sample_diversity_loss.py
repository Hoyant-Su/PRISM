import torch
import torch.nn.functional as F

def sample_diversity_loss(min_encodings, batch_size, n_e):
    sample_histograms = []
    num_vectors_per_sample = min_encodings.shape[0] // batch_size
    for i in range(batch_size):
        start_idx = i * num_vectors_per_sample
        end_idx = (i + 1) * num_vectors_per_sample
        sample_encodings = min_encodings[start_idx:end_idx]
        sample_histogram = torch.mean(sample_encodings, dim=0)
        sample_histograms.append(sample_histogram.unsqueeze(0))
    sample_histograms = torch.cat(sample_histograms, dim=0)

    mean_histogram = torch.mean(sample_histograms, dim=0, keepdim=True)

    loss_kl_div = F.kl_div(
        sample_histograms.log_softmax(dim=1),
        mean_histogram.softmax(dim=1),
        reduction='batchmean'
    )
    return 1 - loss_kl_div

