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
import sklearn.utils
import evaluate

# Testing
import numpy as np
from torchvision import transforms, datasets
import metrics
import sklearn.metrics

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Set up directory for experiment
    print(sys.argv)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "diabetic-retinopathy-detection"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "vit_small_patch16_224_in21k"
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
    num_warm_up_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    
    dataset_name = "_eyePACS_"
    model_directory = os.path.join("runs", model_name + dataset_name + time.strftime("%m_%d_%H_%M_%S"))
    os.mkdir(model_directory)
    print(model_directory)

    # Load datasets split into train, val and test
    dataset_names = ["train", "val", "test"]    
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, random_state=13, max_length=3000)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    datasets["train"].augment=True

    # Setup dataloaders
    batch_size = 512
    mini_batch_size= 64#100
    accumulation_steps = int(batch_size/mini_batch_size)
    num_workers = 4
    dataset_sizes = {name: len(datasets[name]) for name in dataset_names}                  
    dataloaders = {name: torch.utils.data.DataLoader(datasets[name], batch_size=mini_batch_size,
                                            shuffle=False, num_workers=num_workers)
                        for name in ["val", "test"]}   
    dataloaders["train"] = torch.utils.data.DataLoader(datasets["train"], batch_size=mini_batch_size, shuffle=True, num_workers=num_workers, drop_last = True)                     

    # Calc class inbalance and so loss function weights
    data_train_labels = np.array(datasets["train"].get_labels())
    loss_weights = sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.unique(data_train_labels), y=data_train_labels)
    loss_weights = torch.tensor(loss_weights).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names)).to(device)

    num_batches_per_train_epoch = len(datasets["train"]) / batch_size
    num_epochs = int(num_steps//num_batches_per_train_epoch)
    warmup_steps = int(num_warm_up_steps//num_batches_per_train_epoch)
    num_epochs_to_converge = 100
    grad_clip_norm = 1

    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = LRSchedules.WarmupCosineSchedule(optimizer, num_steps, num_warm_up_steps)
    
    # Init tensorboard
    writer = SummaryWriter(model_directory)
    # Add input images to tensorboard for sanity check
    fig = visualisation.sample_batch(dataloaders["train"], class_names)
    writer.add_figure('Input/train', fig)
    fig = visualisation.sample_batch(dataloaders["val"], class_names)
    writer.add_figure('Input/val', fig)

    # # Main training loop
    model, best_loss = train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, len(class_names), writer, model_directory, warmup_steps, num_epochs_to_converge, accumulation_steps, grad_clip_norm)

    # Add sample inference outputs to tensorboard
    fig = visualisation.sample_batch(dataloaders["train"], class_names, model, device)
    writer.add_figure('Inference/train', fig)
    fig = visualisation.sample_batch(dataloaders["val"], class_names, model, device)
    writer.add_figure('Inference/val', fig)

    writer.close()

    evaluate.evaluate_models([model], device, dataloaders["val"], datasets["val"].get_labels(), mini_batch_size, model_directory, "val", [model_name])
    evaluate.evaluate_models([model], device, dataloaders["test"], datasets["test"].get_labels(), mini_batch_size, model_directory, "test", [model_name])
