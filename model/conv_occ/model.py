#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

import os 
from pathlib import Path
import time 

from model import base_pl
from model.archs.encoders.conv_pointnet import ConvPointnet
from model.archs.decoders.conv_occ_arch import ConvOccArch

from utils import mesh, evaluate

class ConvOccNet(base_pl.Model):
    def __init__(self, specs):
        super().__init__(specs)
        
        encoder_specs = self.specs["EncoderSpecs"]
        self.latent_size = encoder_specs["latent_size"]
        self.latent_hidden_dim = encoder_specs["hidden_dim"]
        self.unet_kwargs = encoder_specs["unet_kwargs"]
        self.plane_resolution = encoder_specs["plane_resolution"]

        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]

        lr_specs = self.specs["LearningRate"]
        self.lr_init = lr_specs["init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        self.build_model()


    def build_model(self):
        self.encoder = ConvPointnet(c_dim=self.latent_size, hidden_dim=self.latent_hidden_dim, 
                                        plane_resolution=self.plane_resolution,
                                        unet=(self.unet_kwargs is not None), unet_kwargs=self.unet_kwargs)
        
        self.decoder = ConvOccArch(c_dim=self.latent_size, hidden_size=self.decoder_hidden_dim)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]

 
    # convert sdf to occupancy!
    def training_step(self, x, batch_idx):

        pc = x['point_cloud'].float()
        xyz = x['sdf_xyz'].float()
        gt_sdf = x['gt_sdf'].float()
        gt_occ = (gt_sdf > 0).float()

        # shape latent code
        shape_vecs = self.encoder(pc, xyz)

        logits = self.decoder(xyz, shape_vecs) # output is passed through sigmoid in loss function

        #print("logits shape: ", logits.shape, logits[:,0:10])
        #print("gt occ shape: ", gt_occ.shape, gt_occ[:,0:10])

        #p_r = torch.distributions.Bernoulli(logits=logits)
        # this line is included in the code but is used as logits = p_r.logits; and not sampled; not sure about usage

        loss = F.binary_cross_entropy_with_logits(logits, gt_occ, reduction='none')
        #print("loss shape: ", loss.shape)
        loss = loss.sum(-1).mean()
        
        return loss
        
    def reconstruct(self, model, test_data, eval_dir, testopt=True):
        recon_samplesize_param = 256
        recon_batch = int(2 ** 19)

        gt_pc = test_data['point_cloud'].float()
        print("gt pc shape: ",gt_pc.shape)
        sampled_pc = gt_pc[:,torch.randperm(gt_pc.shape[1])[0:3000]]
        print("sampled pc shape: ",sampled_pc.shape)

        if testopt:
            model = self.pc_opt(model, sampled_pc.cuda())
        model.eval() 
        

        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            mesh_name = test_data["mesh_name"]

            #try:           test_data["indices"]
            mesh.create_mesh_conv(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch)
            #mesh.create_mesh_clean(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch)
            #except Exception as e:
            #    print(e)
            try:
                evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name)
            except Exception as e:
                print(e)

        
        
    def forward(self, pc, query):
        shape_vecs = self.encoder(pc, query)
        decoder_input = torch.cat([shape_vecs, query], dim=-1)
        pred_sdf = self.decoder(decoder_input)

        return pred_sdf

