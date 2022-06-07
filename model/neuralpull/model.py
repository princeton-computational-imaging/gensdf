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
from model.archs.decoders.neuralpull_arch import NeuralPullArch

from utils import mesh, evaluate

class NeuralPull(base_pl.Model):
    def __init__(self, specs, num_objects):
        super().__init__(specs)
        
        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]
        self.geo_init = decoder_specs["geo_init"]

        lr_specs = self.specs["LearningRate"]
        self.lr_init = lr_specs["init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        self.num_objects = num_objects # len(dataset)

        self.build_model()


    def build_model(self):
    
        self.decoder = NeuralPullArch(self.decoder_hidden_dim, geo_init=self.geo_init,
                                      latent_in=self.num_objects)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]

 
    # context and queries from labeled, unlabeled data, respectively 
    def training_step(self, x, batch_idx):

        indices = x['indices']
        xyz = x['sdf_xyz'].float().squeeze()
        gt_point = x['gt_pt'].float().squeeze()

        #print("xyz, gt pt shape: ",xyz.shape, gt_point.shape)

        xyz.requires_grad = True 

        # in auto-decoder framework, an embedding is created and updated
        # in neuralpull, a torch vector is created and a one layer mlp learns the shape feature
        # based on the index in the torch vector 
        # due to this design, batch size can only be 1 
        shape_feature = torch.zeros((xyz.shape[0], self.num_objects), device=self.device)
        shape_feature[:, indices[0]] = 1
        #print("shape feature shape: ",shape_feature.shape)
        pred_sdf = self.decoder(shape_feature, xyz)

        pred_sdf.sum().backward(retain_graph=True)
        grad = xyz.grad.detach()
        grad = F.normalize(grad, dim=-1)
        pred_point = xyz - grad * pred_sdf

        return F.mse_loss(pred_point, gt_point)
        
    def reconstruct(self, model, test_data, eval_dir):
        recon_samplesize_param = 256
        recon_batch = int(2 ** 16)

        gt_pc = test_data['point_cloud'].float().squeeze()
        print("gt pc shape: ",gt_pc.shape)
        #sampled_pc = gt_pc[torch.randperm(gt_pc.shape[0])[0:recon_batch]]
        #print("sampled pc shape: ",sampled_pc.shape)
        model.eval() 
        

        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            mesh_name = test_data["mesh_name"]

            #try:           
            mesh.create_mesh_np(model, test_data["indices"], mesh_filename, recon_samplesize_param, recon_batch)
            #mesh.create_mesh_clean(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch)
            #except Exception as e:
            #    print(e)
            try:
                evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name)
            except Exception as e:
                print(e)

    def forward(self, mesh_idx, query):
        shape_code = torch.zeros((query.shape[0], self.num_objects))
        shape_code[:,mesh_idx] = 1
        pred_sdf = self.decoder( shape_code, query )

        return pred_sdf

        
        