#!/usr/bin/env python3

#import numpy as np
import time 
import logging
import os
#import random
import torch
import torch.utils.data
from . import base 

import pandas as pd 
import csv
import numpy as np 

'''
resplit every 1000 epochs (ref_split_class() is called in train.py callback)
classes are split in half, 1/2 as labeled 1/2 as unlabeled
'''

class MetaSplitDataset(base.Dataset): 

    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        unlab_gen_filepath= "preprocessed/labeled_dataset_gt",

        samples_per_mesh=130000,
        samples_per_batch=16384, # how many samples per batch (randomly sample from samples_per_mesh)
        pc_size=5000, # pc for generating shape code 

        load_files=False,
        save_files=True,
        save_dir = "preprocessed/meta_learning_stage"
    ):


        self.samples_per_mesh = samples_per_mesh
        self.samples_per_batch = samples_per_batch
        self.pc_size = pc_size
        self.unlab_gen_filepath = unlab_gen_filepath
        self.num_iters = int(samples_per_mesh / samples_per_batch)

        self.get_instance_filenames(data_source, split_file)
        print("number of meshes: ",len(self.unlab_gen_files))

        self.query_idx, self.context_idx = self.split_class()

        if load_files:
            print("Loading from saved data...")
            prep_data = torch.load("{}/prep_data.pt".format(save_dir))
            self.point_clouds = prep_data["point_cloud"]
            self.sdf_xyz = prep_data["sdf_xyz"]
            self.gt_pt = prep_data["gt_pt"]
            self.gt_sdf = prep_data["gt_sdf"]

        else:
            self.sdf_xyz      = torch.empty(size=(len(self.unlab_gen_files), samples_per_mesh, 3))
            self.gt_sdf       = torch.empty(size=(len(self.unlab_gen_files), samples_per_mesh, 1))
            self.point_clouds = torch.empty(size=(len(self.unlab_gen_files), pc_size, 3))
            self.gt_pt        = torch.empty(size=(len(self.unlab_gen_files), samples_per_mesh, 3))

            self.preprocess_data()

            if save_files:
                print("Saving files...")
                os.makedirs(save_dir, exist_ok=True)
                torch.save( {
                            "point_cloud":self.point_clouds,
                            "sdf_xyz":self.sdf_xyz,
                            "gt_pt":self.gt_pt,
                            "gt_sdf":self.gt_sdf,
                            },
                            "{}/prep_data.pt".format(save_dir))

        print("shapes: ", self.point_clouds.shape, self.sdf_xyz.shape)

    def __len__(self):
        return int(self.num_classes/2)

 
    def __getitem__(self, idx):

        context_class = int(self.context_idx[idx])
        query_class = int(self.query_idx[idx])

        # from class, randomly sample a mesh inside the class 
        # each mesh is represented by a unique index
        # after getting the index, simply return self.sdf_xyz[index]
        pc, xyz, gt_sdf, _ = self.sample_from_class(context_class, context=True) 
        q_pc, q_xyz, q_gt_sdf, q_gt_pt = self.sample_from_class(query_class, context=False)

        return {
                "context":{
                            "point_cloud":pc,
                            "sdf_xyz": xyz,
                            "gt_sdf": gt_sdf
                        },

                "query":{
                            "point_cloud":q_pc,
                            "sdf_xyz": q_xyz,
                            "gt_sdf": q_gt_sdf,
                            "gt_pt": q_gt_pt
                        }
                }


    def sample_from_class(self, shape_class, context, mesh_iter=None):
    
        mesh_list = self.class_dict[shape_class]
        mesh = torch.randint(low=mesh_list[0], high=mesh_list[-1], size=(1,))[0] # mesh index

        if context:
            f = self.lab_gen_files[mesh]
            pc, xyz, gt_sdf = self.labeled_sampling(f, self.samples_per_batch)
            gt_pt = None

        else:
            samp_idx = torch.randperm(self.samples_per_mesh)[0:self.samples_per_batch]
            pc_idx = torch.randperm(5000)[0:self.pc_size]
            # pc mesh_iter is just repeat, could remove [mesh_iter] and not repeat as well 
            pc = self.point_clouds[mesh][pc_idx].float().squeeze() 
            xyz = self.sdf_xyz[mesh][samp_idx].float().squeeze()
            gt_pt = self.gt_pt[mesh][samp_idx].float().squeeze()
            gt_sdf = self.gt_sdf[mesh][samp_idx].float().squeeze()

        return pc, xyz, gt_sdf, gt_pt


    def preprocess_data(self):

        # sample from self.unlab_gen_files; i.e. the file with points from sampled grid
        print("Loading queries and gt sdf...")
        for idx, i in enumerate(self.unlab_gen_files):
            f=pd.read_csv(i, sep=',',header=None).values

            self.sdf_xyz[idx] = torch.from_numpy(f[:,0:3]).float()
            self.gt_sdf[idx]  = torch.from_numpy(f[:,3]).float().unsqueeze(-1)

        # sample from lab_gen_files; i.e. the original file with point cloud points 
        print("Sampling point cloud and nearest neighbors...")
        for idx, i in enumerate(self.lab_gen_files):
            f=pd.read_csv(i, sep=',',header=None).values

            pc = self.sample_pointcloud(f)  
            nearest_neighbors, _ = self.find_nearest_query_neighbor(pc, self.sdf_xyz[idx])

            self.point_clouds[idx] = pc
            self.gt_pt[idx] = nearest_neighbors

        print("Preprocessed all {} meshes...".format(len(self.unlab_gen_files)))

    def sample_pointcloud(self, f):
        f = f[f[:,-1]==0][:,:3]
        f = torch.from_numpy(f)

        #if f.shape[0] < self.pc_size:
        pc_idx = torch.randperm(f.shape[0])[0:self.pc_size]

        return f[pc_idx].float()


    # the closest point in the pc for all query points 
    def find_nearest_query_neighbor(self, pc, query_points):

        dists = torch.cdist(query_points, pc)
        min_dist, min_idx = torch.min(dists, dim=-1)  
        nearest_neighbors = pc[min_idx]

        return nearest_neighbors, min_dist.unsqueeze(-1)


    # gets the classes, meshes, indices (just the filepaths, no data here)
    def get_instance_filenames(self, data_source, split):
        self.class_dict = {} # dict with class name as keys and mesh indices list as values 
        self.lab_gen_files = []
        self.unlab_gen_files = []
        self.num_classes = 0
        file_index = 0
        for dataset in split: # "acronym"
            for class_name in split[dataset]:
                class_list = []
                for instance_name in split[dataset][class_name]:
                    instance_filename = os.path.join(data_source, dataset, class_name, instance_name, "sdf_data.csv")
                    if not os.path.isfile(instance_filename):
                        logging.warning("Requested non-existent file '{}'".format(instance_filename))
                        continue
                    #class_list.append(instance_filename) # mesh names as values 
                    class_list.append(file_index) # mesh indices as values 
                    file_index += 1 
                    self.lab_gen_files.append(instance_filename) # this contains the original sampled point clouds and query data
                    self.unlab_gen_files.append( os.path.join(self.unlab_gen_filepath, class_name, instance_name, "sdf_gt.csv")  ) # this contains the queries and gt sdf from the grid sampled points 

                self.class_dict[self.num_classes] = class_list
                self.num_classes += 1


    def split_class(self):
        class_idx = torch.randperm(self.num_classes)
        query_idx = class_idx[:int(self.num_classes/2)]
        context_idx = class_idx[int(self.num_classes/2):]
        return query_idx, context_idx

    def ref_split_class(self):
        class_idx = torch.randperm(self.num_classes)
        self.query_idx = class_idx[:int(self.num_classes/2)]
        self.context_idx = class_idx[int(self.num_classes/2):]
    