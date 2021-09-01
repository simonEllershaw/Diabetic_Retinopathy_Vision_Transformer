import os
import argparse
import numpy as np
import torch

from utilities import models
from evaluation.evaluate import evaluate_model
from datasets.eyePACS import EyePACS_Dataset
from datasets.messidor import Messidor_Dataset

def parse_saved_model_dir_path(model_dir):
    root, model_settings_fname = os.path.split(model_dir)#[-1].split("-")
    if "ViT-S" not in model_settings_fname and "ResNet50" not in model_settings_fname:
        _, model_settings_fname = os.path.split(root)
    model_settings = model_settings_fname.split("-")
    model_name = "ViT-S" if model_settings[0] == "ViT" else model_settings[0]
    pretraining = model_settings[-2]
    img_size = int(model_settings[-1])
    return model_name, pretraining, img_size

def load_eyePACs_dataset(data_directory, random_state, use_inception_norm=False, img_size=224):
    # Load eyePACs dataset
    dataset_names = ["train", "val", "test"]    
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    eyePACS_dataset = EyePACS_Dataset(data_directory, random_state=13, use_inception_norm=use_inception_norm, img_size=img_size)
    eyePACS_datasets = eyePACS_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    return eyePACS_datasets

def load_messidor_dataset(data_directory, use_inception_norm=False, img_size=224):
    # Load Messidor-1 dataset
    messidor_datasets = {}
    messidor_datasets["test"] = Messidor_Dataset(data_directory, use_inception_norm=use_inception_norm, img_size=img_size)
    return messidor_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Entry into main training script')
    parser.add_argument('-d', '--model_directory', type = str, default="models")
    parser.add_argument('-e', '--data_directory_eyePACs', type = str, default=os.path.join("data", "eyePACs"))
    parser.add_argument('-m', '--data_directory_messidor', type = str, default=os.path.join("data", "messidor"))
    parser.add_argument('-r', '--random_state', type = int, default=13)

    args = parser.parse_args()
    print(args)

    for root, dirs, files in os.walk(args.model_directory, topdown=False):
        for name in files:
            if name == "model_params.pt" and "21k" in root and "384" in root:
                print(root)
                # Load model
                model_name, pretraining, img_size = parse_saved_model_dir_path(root)
                model_fpath = os.path.join(root, name)
                model, use_inception_norm = models.load_model(model_name, pretraining, 2, img_size, model_fpath)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                # Calc eyePACs performance
                eyePACS_datasets = load_eyePACs_dataset(args.data_directory_eyePACs, args.random_state, use_inception_norm, img_size)
                evaluate_model(model, device, root, eyePACS_datasets, "val")
                evaluate_model(model, device, root, eyePACS_datasets, "test")
                # Calc messidor performance
                messidor_datasets = load_messidor_dataset(args.data_directory_messidor, use_inception_norm, img_size)
                evaluate_model(model, device, root, messidor_datasets, "test")

