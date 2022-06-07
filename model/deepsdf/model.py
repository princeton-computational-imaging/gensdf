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
from model.archs.encoders.auto_decoder import AutoDecoder
from model.archs.decoders.deepsdf_arch import DeepSdfArch

from utils import mesh, evaluate

class DeepSDF(base_pl.Model):
    def __init__(self, specs, num_objects):
        super().__init__(specs)
        
        encoder_specs = self.specs["EncoderSpecs"]
        self.latent_size = encoder_specs["latent_size"]
        
        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]
        self.skip_connection = decoder_specs["skip_connection"]
        self.geo_init = decoder_specs["geo_init"]
        self.weight_norm = decoder_specs["weight_norm"]
        self.tanh_act = decoder_specs["tanh_act"]
        self.dropout = decoder_specs["dropout_prob"]

        lr_specs = self.specs["LearningRate"]
        self.lr_enc_init = lr_specs["enc_init"]
        self.lr_dec_init = lr_specs["dec_init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        self.num_objects = num_objects # len(dataset)
        self.samples_per_mesh = self.specs["SampPerMesh"]
        self.alpha = self.specs["Alpha"]

        self.build_model()

        self.save_hyperparameters()


    def build_model(self):
        
        ad = AutoDecoder(self.num_objects, self.latent_size)
        self.encoder = ad.build_model()
        
        self.decoder = DeepSdfArch(self.latent_size, self.decoder_hidden_dim, geo_init=self.geo_init, 
                                  skip_connection=self.skip_connection, weight_norm=self.weight_norm,
                                  tanh_act=self.tanh_act)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": self.lr_enc_init # 1e-3
                },
                {
                    "params": self.decoder.parameters(),
                    "lr": self.lr_dec_init # 1e-5*batch size
                }
            ]
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]

 
    # context and queries from labeled, unlabeled data, respectively 
    def training_step(self, x, batch_idx):

        indices = x['indices']
        xyz = x['sdf_xyz'].float()
        gt_sdf = x['gt_sdf'].float()

        #print("xyz, gt shape: ",xyz.shape, gt_sdf.shape)

        # one index for each object
        shape_vecs = self.encoder(indices) 
        # each object embedding is 1x256, expand to 16384x256 to concat with xyz

        shape_vecs = shape_vecs.repeat_interleave(self.samples_per_mesh, dim=0)

        xyz = xyz.reshape(xyz.shape[0]*xyz.shape[1],-1)
        gt_sdf = gt_sdf.reshape(gt_sdf.shape[0]*gt_sdf.shape[1],-1)
        # shape_vecs shape: B*16384,256; xyz: B*16384,3
        decoder_input = torch.cat([shape_vecs, xyz], dim=-1).float()
        pred_sdf = self.decoder(decoder_input)
        pred_sdf = torch.clamp(pred_sdf, -0.1, 0.1)
        gt_sdf = torch.clamp(gt_sdf, -0.1, 0.1)
        
        l1_loss = F.l1_loss(pred_sdf, gt_sdf)

        # regularization loss
        l2_size_loss = torch.sum(torch.norm(shape_vecs, dim=-1))
        reg_loss = (self.alpha * min(1, self.current_epoch/100) * l2_size_loss) / gt_sdf.shape[0]  

        loss_dict =  {
                        "reg": reg_loss.detach(),
                        "l1": l1_loss.detach()
                    }
        self.log_dict(loss_dict, prog_bar=True)
        
        return l1_loss +reg_loss


    def forward(self, shape_feature, query):
        shape_vecs = shape_feature.expand(query.shape[0], -1)
        decoder_input = torch.cat([shape_vecs, query], dim=-1).float()
        pred_sdf = self.decoder(decoder_input)

        return pred_sdf


    def reconstruct(self, model, test_data, eval_dir, do_recon=True, do_evaluate=True):

        gt_pc = test_data['point_cloud'].float()
        
        l, latent = self.deepsdf_opt(model, 800, 256, test_data["xyz"], test_data["gt_sdf"])

        print("loss: ",l)
        model.eval() 

        recon_samplesize_param = 256
        recon_batch = 1000000

        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            mesh_name = test_data["mesh_name"]

            if do_recon:
                mesh.create_mesh_default(model, test_data["indices"], mesh_filename, recon_samplesize_param, recon_batch)
               
            if do_evaluate: # chamfer distance
                try:
                    evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name)
                except Exception as e:
                    print(e)


    def deepsdf_opt(
        self,
        decoder,
        num_iterations,
        latent_size,
        sdf_xyz,
        gt_sdf,
        num_samples=30000,
        lr=5e-4,
    ):
        
        latent = torch.ones(1, latent_size).normal_(mean=0, std=0.1).cuda()
        latent.requires_grad = True

        optimizer = torch.optim.Adam([latent], lr=lr)

        loss_l1 = torch.nn.L1Loss()

        for e in range(num_iterations):

            decoder.eval()
            
            xyz = sdf_xyz.cuda().squeeze()
            sdf_gt = gt_sdf.cuda().squeeze()

            sdf_gt = torch.clamp(sdf_gt, -0.1,0.1)

            optimizer.zero_grad()

            latent_inputs = latent.expand(xyz.shape[0], -1)

            inputs = torch.cat([latent_inputs, xyz], -1).cuda().float()

            pred_sdf = decoder.decoder(inputs)

            if e == 0:
                pred_sdf = decoder.decoder(inputs)

            pred_sdf = torch.clamp(pred_sdf, -0.1,0.1).squeeze()

            loss = loss_l1(pred_sdf, sdf_gt)

            loss.backward()
            optimizer.step()

            loss_num = loss.cpu().data.numpy()

        return loss_num, latent


        