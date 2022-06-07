#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import sys
import torch.nn.init as init
import numpy as np


class SirenArch(nn.Module):
    def __init__(self, latent_size=256, hidden_dim=256, weight_norm=False, 
                 skip_connection=False, dropout_prob=0.0, tanh_act=False,
                 geo_init=False, first_init=True
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.skip_connection = skip_connection
        self.dropout_prob = dropout_prob
        self.tanh_act = tanh_act
        self.first_layer_sine_init = first_init

        
        self.block1 = nn.Sequential(
            nn.Linear(self.latent_size+3, hidden_dim),
            Sine(w0=30)
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, 1)
        )


        if self.first_layer_sine_init:
            for m in self.block1.modules():
                with torch.no_grad():
                    if hasattr(m, 'weight'):
                        num_input = m.weight.size(-1)
                        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                        m.weight.uniform_(-1 / num_input, 1 / num_input)


    def forward(self, x):
        out1 = self.block1(x)
        out = self.block2(out1)
        return out.squeeze()


    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)

class Sine(nn.Module):
    def __init__(self, w0 = 1): # pg. 5
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

