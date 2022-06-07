#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np 
import trimesh


import os
import json
import time
from tqdm import tqdm

# remember to add paths in model/__init__.py for new models
from model import *



def main():
    
    model = init_model(specs["Model"], specs, 1)
    
    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
     
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

    file_ext = args.file[-4:]
    if file_ext == ".csv":
        f = pd.read_csv(args.file, sep=',',header=None).values
        f = f[f[:,-1]==0][:,:3]
    elif file_ext == ".ply":
        f = trimesh.load(args.file).vertices
    else:
        print("add your extension type here! currently not supported...")
        exit()


    sampled_points = 15000 # load more points for more accurate reconstruction 
    
    # recenter and normalize
    f -= np.mean(f, axis=0)
    bbox_length = np.sqrt( np.sum((np.max(f, axis=0) - np.min(f, axis=0))**2) )
    f /= bbox_length

    f = torch.from_numpy(f)[torch.randperm(f.shape[0])[0:sampled_points]].float().unsqueeze(0)
    model.load_state_dict(checkpoint['state_dict'])
    model.reconstruct(model, {'point_cloud':f, 'mesh_name':"loaded_file"}, eval_dir="single_recon", testopt=True, sampled_points=sampled_points) 


def init_model(model, specs, num_objects):
    if model == "DeepSDF":
        return DeepSDF(specs, num_objects).cuda()
    elif model == "NeuralPull":
        return NeuralPull(specs, num_objects).cuda()
    elif model == "ConvOccNet":
        return ConvOccNet(specs).cuda()
    elif model == "GenSDF":
        return GenSDF(specs, None).cuda()
    else:
        print("model not loaded...")

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        default="config/gensdf/semi",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default="last",
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--outdir", "-o",
        required=True,
        help="output directory of reconstruction",
    )

    arg_parser.add_argument(
        "--file", "-f",
        required=True,
        help="input point cloud filepath, in csv or ply format",
    )


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
