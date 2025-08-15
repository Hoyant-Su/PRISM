import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

import sys
sys.path.append("ssl_pretraining/utils")
from prompt_attn_visual import visualize_and_save_attention

class DINO_Cine_Projector(nn.Module):
    def __init__(self, cfg, student_model, routing_module, mode="training", log_dir=None):
        super(DINO_Cine_Projector, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.log_dir = log_dir

        self.use_feature_output = cfg.get("use_feature_output", False)
        self.student = student_model.to(self.device)
        self.routing_module = routing_module.to(self.device)
        self.attention_embed_dim = cfg["embed_dim"][-1]

        self.align_dim = cfg.get("align_dim", 30)
        self.student_feature_dim = cfg["embed_dim"][-1]

        num_cross_attn_heads = cfg.get("cross_attn_num_heads", 2)
        self.cross_attn = nn.MultiheadAttention(self.attention_embed_dim, num_cross_attn_heads)

        self.post_img_linear = nn.Linear(self.attention_embed_dim, self.align_dim)
        self.norm_img = nn.LayerNorm(self.align_dim)

        self.student_ref_projector = nn.Linear(self.student_feature_dim, self.align_dim)
        self.norm_img_ref = nn.LayerNorm(self.align_dim)

        if self.student_feature_dim != self.attention_embed_dim:
            self.student_kv_projector = nn.Linear(self.student_feature_dim, self.attention_embed_dim)
        else:
            self.student_kv_projector = nn.Identity()

        bert_model_path = cfg.get("bert_model_path", "../TextEncoder/BERT/bert_model")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        
        bert_output_dim = cfg.get("bert_output_dim", 1024) 

        self.prompt_linear = nn.Linear(bert_output_dim, self.attention_embed_dim)
        self.max_prompt_len = cfg.get("max_prompt_len", 64)

        self.ehr_embed_num = cfg.get("ehr_embed_categories", [3, 3, 3, 3, 3, 3, 3, 3, 3, 10, 7, 9, 9, 8, 11, 6, 6, 11, 11, 11, 5, 3, 6, 5, 4, 3, 5, 10, 7, 4, 5, 5, 10, 3, 11, 7, 11, 10])
        self.ehr_embeddings = nn.ModuleList([
            nn.Embedding(embed_num, self.align_dim)
            for embed_num in self.ehr_embed_num
        ])
        self.norm_ehr = nn.LayerNorm(self.align_dim, elementwise_affine=False)

        self.ehr_indexer = cfg.get("ehr_indexer", {
            'Clinical': [0, 1, 2, 22, 23, 24, 25, 29, 21, 28, 30, 31, 32],
            'Biochemical': [14, 15, 16, 17, 18, 19, 20], 
            'Physiological': [9, 8, 10, 11, 12, 26, 27, 33, 34, 35, 36, 37], 
            'Pharmacological': [3, 4, 5, 6, 7]
        })
        
        for embedding_layer in self.ehr_embeddings:
            embedding_layer.weight.requires_grad = False



    def forward(self, x_temporal, x_spatial, prompt_text, ehr_feats, idx, case_id):
        batch_size = x_temporal.size(0)

        features_temporal, attn_temporal = self.student(x_temporal)[-2], self.student(x_temporal)[-1]
        features_temporal = features_temporal.flatten(2).permute(0, 2, 1)        
        pooled_student_feature = features_temporal.mean(dim=1)

        img_token_ref_unnormed = self.student_ref_projector(pooled_student_feature.detach())
        img_token_reference = self.norm_img_ref(img_token_ref_unnormed) # (B, self.align_dim)

    
        encoded = self.tokenizer(prompt_text, padding='max_length', max_length=self.max_prompt_len, 
                                 truncation=True, return_tensors='pt')
        prompt_input_ids = encoded['input_ids'].to(self.device)
        prompt_attn_mask = encoded['attention_mask'].to(self.device)
        
        prompt_bert_output, routing_weight = self.routing_module(prompt_input_ids, prompt_attn_mask)
        
        query_tokens_for_attn = self.prompt_linear(prompt_bert_output) 
        Q = query_tokens_for_attn.permute(1, 0, 2)
        kv_tokens_for_attn = self.student_kv_projector(features_temporal)
        K = kv_tokens_for_attn.permute(1, 0, 2)
        V = K 
        attn_output_raw, attn_weights = self.cross_attn(Q, K, V, attn_mask=None, key_padding_mask=None, need_weights=True, average_attn_weights=False)
        
        
        if self.mode != "training":
            attn_weights = attn_weights.mean(dim=1)  # (B, Q_len, K_len)

            student_out = self.student(x_temporal)[-2]  # (B, C, D_out, H_out, W_out)
            #student_out = self.student(x_temporal)[-1]  # (B, C, D_out, H_out, W_out)
            D_out, H_out, W_out = student_out.shape[-3:]
            features_temporal = student_out.flatten(2).permute(0, 2, 1)

            visualize_and_save_attention(
                attn_weights=attn_weights,
                save_dir=f'{self.log_dir}',
                input_shape=(x_temporal.shape[2], x_temporal.shape[3], x_temporal.shape[4]),  # D, H, W
                patch_shape=(D_out, H_out, W_out),
                original_image=x_temporal,
                prefix='cross_attn',
                batch_idx=0,
                step=idx,
                prompt = prompt_text[0],
                case_id = case_id[0]
            )

        attn_output_permuted = attn_output_raw.permute(1, 2, 0) 
        original_query_permuted = query_tokens_for_attn.permute(0, 2, 1)

        z_unpooled = torch.cat([original_query_permuted, attn_output_permuted], dim=-1) 
        z_features_pooled = z_unpooled.mean(dim=-1) # (B, self.attention_embed_dim)
      
        img_token_aligned = self.post_img_linear(z_features_pooled) # (B, self.align_dim)
        img_token_aligned = self.norm_img(img_token_aligned)

        ehr_feats_squeezed = ehr_feats.squeeze(1) # (B, num_ehr_features_total)
        ehr_group_contributions = [] 
        for group_name, group_indices in self.ehr_indexer.items():
            group_feature_embeddings = []
            if not group_indices: continue 

            for feature_idx_in_original_ehr in group_indices:
                embedding = self.ehr_embeddings[feature_idx_in_original_ehr](
                    ehr_feats_squeezed[:, feature_idx_in_original_ehr].long()
                ) # (B, self.align_dim)
                group_feature_embeddings.append(embedding)
            current_group_embeddings_stacked = torch.stack(group_feature_embeddings, dim=1)
            current_group_aggregated_embedding = current_group_embeddings_stacked.mean(dim=1) 
            ehr_group_contributions.append(current_group_aggregated_embedding.unsqueeze(1))

        if not ehr_group_contributions:
            ehr_token = torch.zeros(batch_size, self.align_dim, device=self.device, dtype=img_token_aligned.dtype)
        else:
            ehr_group_contributions_cat = torch.cat(ehr_group_contributions, dim=1)
            ehr_token = (ehr_group_contributions_cat * routing_weight.unsqueeze(-1)).sum(dim=1) # (B, self.align_dim)
        
        ehr_token = self.norm_ehr(ehr_token)
        
        if not self.use_feature_output:
            return img_token_aligned, ehr_token, img_token_reference
        else:
            return img_token_aligned, img_token_aligned, img_token_reference