#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import os
import json
import time

# add paths in model/__init__.py for new models
from model import *


def main():
    
    train_dataset = init_dataset(specs["TrainData"], specs)

    # unsupervised methods require sampler; e.g. NeuralPull
    if specs["TrainData"] == "unlabeled":
        from dataloader.unlabeled_ds import Sampler
        sampler = Sampler(train_dataset, len(train_dataset))
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers= 8 if args.workers is None else args.workers,
            drop_last=True,
            sampler=sampler
        )   
        dataloaders = [train_dataloader]

    # GenSDF semi-supervised stage; load two dataloaders
    elif specs["TrainData"] == "semi":
        lab_set, unlab_set = train_dataset

        from dataloader.unlabeled_ds import Sampler
        sampler = Sampler(unlab_set, len(unlab_set))
        unlab_dataloader = torch.utils.data.DataLoader(
            unlab_set,
            batch_size=args.batch_size,
            num_workers= 8 if args.workers is None else args.workers,
            drop_last=True,
            sampler=sampler
        )  
        lab_dataloader = torch.utils.data.DataLoader(
            lab_set,
            batch_size=args.batch_size,
            num_workers= 8 if args.workers is None else args.workers,
            drop_last=True,
            shuffle=True
        )
        dataloaders = {"context":lab_dataloader, "query":unlab_dataloader}

    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers= 8 if args.workers is None else args.workers,
            drop_last=True,
            shuffle=True
        )
        dataloaders = [train_dataloader]

    data_len = len(train_dataset) if specs["TrainData"] != "semi" else len(lab_set)+len(unlab_set)
    print("Training on {} objects...".format(data_len))

    model = init_model(specs["Model"], specs, data_len, dataloaders)

    max_epochs = specs["NumEpochs"]
    log_frequency = specs["LogFrequency"]

    
    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  

    callbacks = []

    if specs["Model"] == "GenSDF" and specs["SplitDataFreq"]:
        # split dataset into two subsets after certain number of epochs
        # for GenSDF, meta-learning stage
        class SplitCallback(Callback):
            def __init__(self, split_every_n_epochs):
                self.split_every_n_epochs = split_every_n_epochs
                self.counter = 0

            def on_train_epoch_end(self, *args, **kwargs):
                if self.counter % self.split_every_n_epochs == 0:
                    train_dataset.ref_split_class()
                self.counter+=1

        split_cb = SplitCallback(split_every_n_epochs=specs["SplitDataFreq"])
        callbacks.append(split_cb)

    callback = ModelCheckpoint(
        dirpath=args.exp_dir, filename='{epoch}',
        save_top_k=-1, save_last=True, every_n_epochs=log_frequency)

    callbacks.append(callback)

    
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, max_epochs=max_epochs, 
                        callbacks=callbacks)
    trainer.fit(model=model, ckpt_path=resume) 



def init_model(model, specs, num_objects, dataloaders):
    if model == "GenSDF":
        return GenSDF(specs, dataloaders)
    elif model == "DeepSDF":
        return DeepSDF(specs, num_objects)
    elif model == "NeuralPull":
        return NeuralPull(specs, num_objects)
    elif model == "ConvOccNet":
        return ConvOccNet(specs)
    else:
        print("model not loaded...")
        exit()

def init_dataset(dataset, specs):

    # GenSDF semi-supervised stage, load two dataloaders for labeled and unlabeled datasets
    if dataset == "semi":
        from dataloader.labeled_ds import LabeledDS
        from dataloader.unlabeled_ds import UnLabeledDS
        labeled_train = specs["LabeledTrainSplit"]
        with open(labeled_train, "r") as f:
            labeled_train_split = json.load(f)
        unlabeled_train = specs["UnLabeledTrainSplit"]
        with open(unlabeled_train, "r") as f:
            unlabeled_train_split = json.load(f)

        return LabeledDS(
            specs["DataSource"], labeled_train_split, 
            samples_per_mesh=specs["LabSamplesPerMesh"], pc_size=specs["LabPCsize"]
            ), UnLabeledDS(specs["DataSource"], unlabeled_train_split, 
                samples_per_mesh=specs["SampPerMesh"], pc_size=specs["PCsize"],
                samples_per_batch=specs["SampPerBatch"])


    train_split_file = specs["TrainSplit"]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    # GenSDF meta-learning stage
    if dataset == "meta":
        from dataloader.meta_ds import MetaSplitDataset
        return MetaSplitDataset(specs["DataSource"], train_split,
                samples_per_batch=specs["SampPerBatch"], pc_size=specs["PCsize"],
                samples_per_mesh=specs["SampPerMesh"])

    # for fully-supervised methods; e.g., DeepSDF, ConvOccNet
    elif dataset == "labeled":
        from dataloader.labeled_ds import LabeledDS
        return LabeledDS(specs["DataSource"], train_split, 
                samples_per_mesh=specs["SampPerMesh"], pc_size=specs.get("PCsize",1024))

    # for fully-unsupervised methods; e.g. NeuralPull, SAL
    elif dataset == "unlabeled":
        from dataloader.unlabeled_ds import UnLabeledDS, Sampler
        return UnLabeledDS(specs["DataSource"], train_split, 
                samples_per_mesh=specs["SampPerMesh"], pc_size=specs["PCsize"],
                samples_per_batch=specs["SampPerBatch"], query_per_point=specs["QueryPerPoint"])


    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default=None,
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--batch_size", "-b",
        default=1, type=int
    )

    arg_parser.add_argument(
        "--workers", "-w",
        default=None, type=int
    )


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
