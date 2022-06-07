#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import json
import os 
from pathlib import Path
import time 

import pandas as pd 
import numpy as np 

from utils import evaluate, mesh


class Model(pl.LightningModule):
    # specs is a json filepath that contains the specifications for the experiment 
    def __init__(self, specs):
        super().__init__()

        if type(specs) == dict:
            self.specs = specs

        else:
            if not os.path.isfile(specs):
                raise Exception("The specifications at {} do not exist!!".format(specs))
            self.specs = json.load(open(specs))

    # forward is used for inference/predictions
    # def forward(self, x):
    #     raise NotImplementedError
    # training_step replaces the original forward function 
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def configure_optimizer(self):
        raise NotImplementedError


    


        
        
