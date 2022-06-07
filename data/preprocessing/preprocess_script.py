import json 
import os 
import shutil 
import argparse 
import subprocess
import numpy as np

# preprocess functions for ShapeNet data 
# def get_meshes_dict(split_file):
#     with open(split_file) as f:
#         data = json.load(f)
#     return data['ShapeNetV2']

# def run_sdf_gen(root_dir, output_dir, split_file, mkdir):
#     data = get_meshes_dict(split_file)
#     for k in data.keys():
#         for v in data[k]:
#             path = os.path.join(root_dir, k, v, "model.obj") 
#             output_path = os.path.join(output_dir, k, v)
#             if mkdir:
#                 os.makedirs(output_path, exist_ok=True)
#             #./sdf_gen $path $output_path || true
#             args = ("build/sdf_gen", path, output_path, "|| true")
#             popen = subprocess.Popen(args, stderr=subprocess.DEVNULL)
#             popen.wait()

# return True if no negative tensors in the csv file
def check_neg(csvpath):
    f = np.loadtxt(csvpath, delimiter=',')
    
    neg_tensors = f[f[:,-1]<0]
    pos_tensors = f[f[:,-1]>0]
    
    print("neg, pos num: ", len(neg_tensors), len(pos_tensors))
    
    if len(neg_tensors)==0:
        return True 
    return False  

def run_sdf_gen(root_dir, class_name):
    class_dir = os.path.join(root_dir, class_name) # root_dir = acronym; class_name, Plant, Spoon..etc

    meshes = os.listdir(class_dir) # in Plant, Spoon, there are multiple .obj files
    meshes = [i for i in meshes if i.endswith(".obj")]

    #split_file = "meshes.csv"
    #f = np.loadtxt(os.path.join(class_dir, split_file), dtype=str)
    
    mesh_folders = []
    for mesh in meshes:
        # create folders of the mesh name and move obj, sdf_gt files into the folder 
        mesh_folder = os.path.join(class_dir, mesh.split(".")[0])
        mesh_folders.append(mesh_folder)
        os.makedirs(mesh_folder, exist_ok=True)
        shutil.move(os.path.join(class_dir, mesh), os.path.join(mesh_folder, "model.obj"))
        
        args = ("build/sdf_gen",  os.path.join(mesh_folder, "model.obj"), mesh_folder, "|| true")
        popen = subprocess.Popen(args, stderr=subprocess.DEVNULL)
        popen.wait()

    # after all csv gt files created, remove those without negative sdv 
    for mesh in mesh_folders:
        csvfile = os.path.join(mesh, "sdf_data.csv")
        if check_neg(csvfile):
            print("removing {} because there are no negative SDV!".format(mesh.split("/")[-1]))
            shutil.rmtree(mesh)



arg_parser = argparse.ArgumentParser()
#arg_parser.add_argument('--split_file', '-s', default='sv2_sofas_train.json')
arg_parser.add_argument('--root_dir', '-r', default='../DeepSDF/data/SdfSamples/acronym')
arg_parser.add_argument('--class_name', '-c', nargs="+")
#arg_parser.add_argument('--output_dir', '-o', default='../DeepSDF/data/SDFSamples/ShapeNetV2')
#arg_parser.add_argument('--mkdir', action='store_true')

args = arg_parser.parse_args()

if args.class_name == ["all"]:
    classes = os.listdir(args.root_dir)
    if ".DS_Store" in classes:
        classes.remove(".DS_Store")
    classes.sort()
    print(classes)
    # for c in classes:
    #     print("processing {}".format(c))
    #     run_sdf_gen(args.root_dir, c)
    for idx in range(start_idx, len(classes)):
        print("processing {}".format(classes[idx]))
        run_sdf_gen(args.root_dir, classes[idx])

else:
    for c in args.class_name:
        print("processing {}".format(c))
        run_sdf_gen(args.root_dir, c)
