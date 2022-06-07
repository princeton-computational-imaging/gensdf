import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

'''
neuralpull.py from github repo, lines 248:263
1. pass shape feature through a layer with output 128
2. pass xyz through a layer with output 512
3. concat 1 and 2, then pass through 8 more layers with 512 dim
4. finally a fully connected layer for predicted value
'''

class NeuralPullArch(nn.Module):
    def __init__(self, hidden_dim=512, geo_init=True, latent_in=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(latent_in, 128),
            nn.ReLU()
        )

        self.block1 = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(512+128, 512),
            nn.ReLU(),
            * ([nn.Linear(512, 512), nn.ReLU()] * 8)
        )

        self.block3 = nn.Sequential(
            nn.Linear(512, 1),
        )
        if geo_init:
            self._init_weight()

    def _init_weight(self):
        for m in self.block2.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(512))
                init.constant_(m.bias, 0.0)
        for m in self.block3.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(512), std=0.000001)
                init.constant_(m.bias, -0.5)

    def forward(self, shape_vec, xyz):
        '''
        batch size (B) has to be 1 for this design 
        shape_vec: (B, 5000, 1)
        xyz: (B, 5000, 3)
        '''
        shape_feature = self.encoder(shape_vec) # # (B, 5000, 128)
        out1 = self.block1(xyz) # (B, 5000, 512)
        block2_input = torch.relu(torch.cat((shape_feature, out1), dim=-1))  # (B, 5000, 640)
        out2 = self.block2(block2_input) # (B, 5000, 512)
        out = self.block3(out2) # (B, 5000, 1)

        return out