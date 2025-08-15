import torch.nn as nn
import torch

class Cine_Survival_Prediction_Head(nn.Module):
    def __init__(self, cfg, student_model):
        super(Cine_Survival_Prediction_Head, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_feature_output = cfg["use_feature_output"]

        self.student = student_model.to(self.device)

        self.cross_attn = nn.MultiheadAttention(cfg["embed_dim"][-1], 2)

        self.norm = nn.LayerNorm(cfg["embed_dim"][-1])
        self.head = nn.Linear(cfg["embed_dim"][-1], cfg["num_time_bins"])

    def CTSL_CrossAttn(self, feat_temporal, feat_spatial):
        Q = feat_temporal.flatten(2).permute(2, 0, 1)
        K = feat_spatial.flatten(2).permute(2, 0, 1)
        V = feat_spatial.flatten(2).permute(2, 0, 1)
        features, attn_weights = self.cross_attn(Q, K, V)
        return features, attn_weights

    def forward(self, x_temporal, x_spatial):
        _, _, _, features_temporal, features_spatial, input_temporal, input_spatial = self.student(x_temporal, x_spatial)

        z_features, z_attention_weights = self.CTSL_CrossAttn(features_temporal, features_spatial)
        input_features, input_attention_weights = self.CTSL_CrossAttn(input_spatial, input_spatial)

        z_features = z_features.permute(1, 2, 0).mean(-1)
        input_features = input_features.permute(1, 2, 0).mean(-1)

        z_logits = self.head(self.norm(z_features))
        input_logits = self.head(self.norm(input_features))

        if not self.use_feature_output:
            return z_logits, input_logits
        else:
            return z_features, z_features