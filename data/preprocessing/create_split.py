import json 
import os 
import shutil 
import argparse 
import subprocess
import numpy as np

# dataset_dir, e.g. ShapeNetV2 or acronym 
# class_names, e.g. Plant, Spoon
# split_ratio: 1 for no testing split, 0.8 for 0.8 training 0.2 testing
def create_split_json(dataset_dir, class_names, output_dir, split_ratio, split_classes):
    train_dic = {}
    test_dic = {}
    dataset_name = dataset_dir.split("/")[-1]
    train_dic[dataset_name] = {}
    test_dic[dataset_name] = {}
    if class_names == ["all"]:
        class_names = os.listdir(dataset_dir)
        if ".DS_Store" in class_names:
            class_names.remove(".DS_Store")
        class_names.sort()
        print(class_names)
    for c in class_names:
        mesh_names = [i for i in os.listdir(os.path.join(dataset_dir, c)) if os.path.isdir(os.path.join(dataset_dir, c, i))]

        if len(mesh_names) < split_classes:
            # directly throw all to test split
            test_dic[dataset_name][c] = []
            for m in mesh_names:
                test_dic[dataset_name][c].append(m)
            continue

        if c not in train_dic[dataset_name]:
            train_dic[dataset_name][c] = []
            test_dic[dataset_name][c] = []
        split_idx = round(len(mesh_names)*split_ratio)
        for i in range(split_idx):
            train_dic[dataset_name][c].append(mesh_names[i])
        for i in range(split_idx, len(mesh_names)):
            test_dic[dataset_name][c].append(mesh_names[i])

    with open("{}/{}_train_split.json".format(output_dir, dataset_name), 'w') as tr:
        json.dump(train_dic, tr, indent=4)
    with open("{}/{}_test_split.json".format(output_dir, dataset_name), 'w') as tt:
        json.dump(test_dic, tt, indent=4)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset_dir', '-d', default='../DeepSDF/data/SdfSamples/acronym')
arg_parser.add_argument('--class_names', '-c', nargs="+", default=["all"])
arg_parser.add_argument('--output_dir', '-o', default='../DeepSDF/examples/splits')
arg_parser.add_argument('--split_ratio', '-r', default=0.9, help="splits within a class")
arg_parser.add_argument('--split_classes', '-s', default=40, help="splits based on classes, min number of meshes required to be in training")

args = arg_parser.parse_args()
create_split_json(args.dataset_dir, args.class_names, args.output_dir, args.split_ratio, args.split_classes)
