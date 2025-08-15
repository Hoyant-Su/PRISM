import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, in_chans, variable_stride):
        super(DecoderBlock, self).__init__()

        self.deconv1 = nn.ConvTranspose3d(embed_dim[3], embed_dim[2], kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose3d(embed_dim[2], embed_dim[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose3d(embed_dim[1], embed_dim[0], kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose3d(embed_dim[0], in_chans, kernel_size=variable_stride, stride=variable_stride)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        z_q, z_e1, z_e2, z_e3 = x

        x4_up = self.relu1(self.deconv1(z_q))
        x3_up = self.relu2(self.deconv2(z_e3 + x4_up))
        x2_up = self.relu3(self.deconv3(z_e2 + x3_up))
        x1_up = (self.deconv4(z_e1 + x2_up))
        hat_x = x1_up

        return hat_x
