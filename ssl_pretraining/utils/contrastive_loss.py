import torch
import torch.nn.functional as F

def contrastive_loss(img_token, ehr_token, temperature=0.07):
    img_token = F.normalize(img_token, dim=1)
    ehr_token = F.normalize(ehr_token, dim=1)

    logits = torch.matmul(img_token, ehr_token.T) / temperature

    batch_size = img_token.size(0)
    targets = torch.arange(batch_size, device=img_token.device)

    loss_i2e = F.cross_entropy(logits, targets)
    loss_e2i = F.cross_entropy(logits.T, targets)

    return (loss_i2e + loss_e2i) / 2
