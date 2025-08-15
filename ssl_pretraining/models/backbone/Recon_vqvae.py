import torch
import torch.nn as nn
from models.backbone.visual_encoder import UniformerEncoder
from models.backbone.quantizer import VectorQuantizer3D
from models.backbone.visual_decoder import DecoderBlock
from utils.sample_diversity_loss import sample_diversity_loss

class VQ_VAE_4D(nn.Module):
    def __init__(self, embed_dim=[64, 128, 320, 728], in_chans_temporal=24, in_chans_spatial=24, variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05):
        super(VQ_VAE_4D, self).__init__()
        self.n_e = n_e
        
        self.encoder = UniformerEncoder(depth=[5, 8, 20, 7], in_chans_temporal=in_chans_temporal, in_chans_spatial=in_chans_spatial, variable_stride=variable_stride)

        self.codebooks = nn.ModuleList([
            VectorQuantizer3D(n_e, e_dim, beta) for _ in range(2)
        ])

        self.decoder_block = nn.ModuleList([
            DecoderBlock(embed_dim=embed_dim, in_chans=in_chans_temporal, variable_stride=variable_stride) for i in range(2)
        ])

    def forward(self, x_temporal, x_spatial):

        z_e1_temporal, z_e2_temporal, z_e3_temporal, z_e4_temporal, _ = self.encoder(x_temporal, "temporal")
        z_e1_spatial, z_e2_spatial, z_e3_spatial, z_e4_spatial, _ = self.encoder(x_spatial, "spatial")

        codebook_temporal_loss, z_q_temporal, perplexity_temporal, min_encodings_temporal, min_encoding_indices_temporal, z_e_temporal = self.codebooks[0](z_e4_temporal)
        codebook_spatial_loss, z_q_spatial, perplexity_spatial, min_encodings_spatial, min_encoding_indices_spatial, z_e_spatial = self.codebooks[1](z_e4_spatial)
        codebook_loss = (codebook_spatial_loss + codebook_temporal_loss) / 2
        perplexity = 2 / (perplexity_spatial + perplexity_temporal)
        codebook_loss += perplexity

        x_temporal = self.decoder_block[0]([z_q_temporal, z_e1_temporal, z_e2_temporal, z_e3_temporal])
        x_spatial = self.decoder_block[1]([z_q_spatial, z_e1_spatial, z_e2_spatial, z_e3_spatial])

        print('---')
        return codebook_loss, x_temporal, x_spatial, z_q_temporal, z_q_spatial, z_e_temporal, z_e_spatial

if __name__ == "__main__":
    B, C, T, H, W = 1, 1, 24, 96, 96
    x_temporal = torch.randn(B, C, T, H, W)
    x_spatial = torch.randn(B, C, T, H, W)
    model = VQ_VAE_4D(embed_dim=[64, 128, 320, 512], in_chans=1, variable_stride=(2,4,4), n_e=128, e_dim=512, beta=0.05)

    codebook_loss, hat_x_temporal, hat_x_spatial = model(x_temporal, x_spatial)

    codebook_params = 0
    decoder_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if 'codebook' in name:
            codebook_params += param.numel()
        elif 'decoder' in name:
            decoder_params += param.numel()
        total_params += param.numel()
    print(total_params,codebook_params, decoder_params)