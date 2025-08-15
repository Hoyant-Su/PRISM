import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalDistanceLoss(nn.Module):
    def __init__(self, margin=0.2, distance_metric='euclidean'):
        super(RelationalDistanceLoss, self).__init__()
        self.margin = margin
        if distance_metric not in ['euclidean', 'cosine']:
            raise ValueError(f"Unsupported distance metric: {distance_metric}. Choose 'euclidean' or 'cosine'.")
        self.distance_metric = distance_metric
        print(f"RelationalDistanceLoss initialized with margin={margin}, distance_metric='{distance_metric}'")

    def _get_pairwise_distances(self, x):
        if self.distance_metric == 'euclidean':

            x_expanded_a = x.unsqueeze(1)
            x_expanded_b = x.unsqueeze(0)

            distances = torch.cdist(x_expanded_a, x_expanded_b, p=2.0).squeeze(0)
            return distances
        
        elif self.distance_metric == 'cosine':
            x_normalized = F.normalize(x, p=2, dim=1)
            similarity_matrix = torch.matmul(x_normalized, x_normalized.transpose(0, 1))
            distances = 1 - similarity_matrix
            distances = torch.clamp(distances, min=0.0)
            return distances

    def forward(self, img_tokens, ehr_tokens):
        batch_size = img_tokens.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=img_tokens.device, requires_grad=True)

        d_ehr_matrix = self._get_pairwise_distances(ehr_tokens)
        d_img_matrix = self._get_pairwise_distances(img_tokens)

        total_loss = 0.0
        num_valid_triplets = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                for k in range(j + 1, batch_size):
                    if k == i:
                        continue

                    d_ehr_ij = d_ehr_matrix[i, j]
                    d_ehr_ik = d_ehr_matrix[i, k]

                    d_img_ij = d_img_matrix[i, j]
                    d_img_ik = d_img_matrix[i, k]

                    if d_ehr_ij < d_ehr_ik:
                        loss_triplet = F.relu(d_img_ij - d_img_ik + self.margin)

                    elif d_ehr_ik < d_ehr_ij:
                        loss_triplet = F.relu(d_img_ik - d_img_ij + self.margin)
                    else:
                        loss_triplet = torch.abs(d_img_ij - d_img_ik)

                    total_loss += loss_triplet
                    num_valid_triplets += 1

        if num_valid_triplets == 0:
            return torch.tensor(0.0, device=img_tokens.device, requires_grad=True)

        return total_loss / num_valid_triplets