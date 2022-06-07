#!/usr/bin/env python3

import argparse
import logging
import json
import numpy as np
import pandas as pd 
import os, sys
import trimesh
from scipy.spatial import cKDTree as KDTree

import csv

def main(gt_pc, recon_mesh, out_file, mesh_name):

    gt_pc = gt_pc.cpu().detach().numpy().squeeze()

    recon_mesh = trimesh.load(os.path.join(os.getcwd(), recon_mesh)+".ply")
    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, gt_pc.shape[0])

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    loss_chamfer = gt_to_recon_chamfer + recon_to_gt_chamfer    

    out_file = os.path.join(os.getcwd(), out_file)

    with open(out_file,"a",) as f:
        writer = csv.writer(f)
        writer.writerow([mesh_name[0],loss_chamfer])


def single_eval(gt_csv, recon_mesh):
    f=pd.read_csv(gt_csv, sep=',',header=None).values
    f = f[f[:,-1]==0][:,:3]
    pc_idx = np.random.choice(f.shape[0], 30000, replace=False)
    gt_pc = f[pc_idx] 

    recon_mesh = trimesh.load( recon_mesh )
    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, 30000)

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    loss_chamfer = gt_to_recon_chamfer + recon_to_gt_chamfer

    print("CD loss: ", loss_chamfer)



if __name__ == "__main__":
    single_eval(sys.argv[1], sys.argv[2])
