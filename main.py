import os
import time
import pickle
import sys

import timm
from torch.utils.data import Dataset 
from torch.utils.data.sampler import WeightedRandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import LRSchedules
from train import train_model
import visualisation
from eyePACS import EyePACS_Dataset

#testing
import numpy as np
from torchvision import transforms, datasets

if __name__ == "__main__":
    # Set up directory for experiment
    model_name = "resnet18"
    dataset_name = "_eyePACS_"
    run_directory = os.path.join("runs", model_name+ dataset_name + time.strftime("%m_%d_%H_%M_%S"))
    os.mkdir(run_directory)

    # Load data
    dataset_names = ["train", "val", "test"]
    data_directory = sys.argv[1]
    dataset = EyePACS_Dataset(data_directory)
    class_names = dataset.class_names

    datasets_split = dataset.split_train_test_val_sets(0.6, 0.2, 0.2)
    datasets_split[0].augment=True

    dataset_indicies = {dataset_names[i]: datasets_split[i].indices for i in range(len(datasets_split))}
    with open(os.path.join(run_directory, "dataset_indexes.pkl"), "wb+") as index_file:
        pickle.dump(dataset_indicies, index_file)

    # # Setup dataloaders
    batch_size= 164
    dataset_sizes = {dataset_names[x]: len(datasets_split[x]) for x in range(len(dataset_names))}                    
    dataloaders = {dataset_names[x]: torch.utils.data.DataLoader(datasets_split[x], batch_size=batch_size,
                                            shuffle=False, num_workers=4)
                        for x in range(len(dataset_names))}   
    dataloaders["train"].shuffle = True
    
    # # Calc class inbalance and so loss function weights
    data_train = datasets_split[0]
    data_train_labels = dataset.labels_df.level.iloc[data_train.indices]
    data_train_class_frequency = torch.tensor(data_train_labels.value_counts(sort=False))
    data_train_class_weights = 1/data_train_class_frequency

    # Set hyperparameters
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    drop_out_rate = 0.5
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names), drop_block_rate=drop_out_rate).to(device)
    criterion = nn.CrossEntropyLoss(weight=data_train_class_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = LRSchedules.WarmupCosineSchedule(optimizer, num_epochs)
    grad_clip_norm = 1

    # Init tensorboard
    writer = SummaryWriter(run_directory)

    # # Add input images to tensorboard for sanity check
    fig = visualisation.sample_batch(dataloaders["train"], class_names)
    writer.add_figure('Input/train', fig)
    fig = visualisation.sample_batch(dataloaders["val"], class_names)
    writer.add_figure('Input/val', fig)

    # # Main training loop
    model, best_acc = train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, len(class_names), writer, run_directory, grad_clip_norm)

    # Add sample inference outputs to tensorboard
    fig = visualisation.sample_batch(dataloaders["train"], class_names, model, device)
    writer.add_figure('Inference/train', fig)
    fig = visualisation.sample_batch(dataloaders["val"], class_names, model, device)
    writer.add_figure('Inference/val', fig)

    writer.close()

