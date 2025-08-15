import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer3D(nn.Module):

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer3D, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.register_buffer("ema_embedding", torch.zeros_like(self.embedding.weight))

    def forward(self, z):
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        batch_size = z.shape[0]
        num_vectors_per_sample = z_flattened.shape[0] // batch_size

        for i in range(batch_size):
            start_idx = i * num_vectors_per_sample
            end_idx = (i + 1) * num_vectors_per_sample

            sample_indices_flat = min_encoding_indices[start_idx:end_idx].squeeze().clone().detach()
            unique_indices_this_sample = torch.unique(sample_indices_flat).tolist()
            num_to_show = min(5, num_vectors_per_sample)
            if num_to_show > 0:
                print(f"  Sample {i} - First {num_to_show} indices (flat): {sample_indices_flat[:num_to_show].tolist()}")

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()

        with torch.no_grad():
            self.ema_embedding.data = self.beta * self.ema_embedding.data + (1 - self.beta) * self.embedding.weight.data

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        return loss, z_q, perplexity, min_encodings, min_encoding_indices, z.permute(0, 4, 1, 2, 3)
