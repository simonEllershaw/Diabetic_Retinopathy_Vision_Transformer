import os
import time

import timm
from torch.utils.data import Dataset

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

dataset = EyePACS_Dataset("config.json")
datasets_split = dataset.split_train_test_val_sets(0.6, 0.2, 0.2)

dataset_names = ["train", "val", "test"]
dataset_sizes = {dataset_names[x]: len(datasets_split[x]) for x in range(len(dataset_names))}                    
dataloaders = {dataset_names[x]: torch.utils.data.DataLoader(datasets_split[x], batch_size=164,
                                        shuffle=False, num_workers=0)
                    for x in range(len(dataset_names))}   
dataloaders["train"].shuffle = True

# # Set up directory to save model data
model_name = "resnet18"
dataset_name = "_eyePACS_"
run_directory = os.path.join("runs", model_name+ dataset_name + time.strftime("%m_%d_%H_%M_%S"))

# Set hyperparameters
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = timm.create_model("resnet18", pretrained=True, num_classes=len(dataset.class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = LRSchedules.WarmupCosineSchedule(optimizer, num_epochs)
grad_clip_norm = 1

# # Init tensorboard
writer = SummaryWriter(run_directory)

# Add input images to tensorboard for sanity check
fig = visualisation.sample_batch(dataloaders["train"], dataset.class_names)
writer.add_figure('Input/train', fig)

fig = visualisation.sample_batch(dataloaders["val"], dataset.class_names)
writer.add_figure('Input/val', fig)

# Main training loop
model, best_acc = train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, writer, run_directory, grad_clip_norm)

# Add sample inference outputs to tensorboard
fig = visualisation.sample_batch(dataloaders["train"], dataset.class_names, model, device)
writer.add_figure('Inference/train', fig)
fig = visualisation.sample_batch(dataloaders["val"], dataset.class_names, model, device)
writer.add_figure('Inference/val', fig)

writer.close()

# model = timm.create_model("resnet18", pretrained=True, num_classes=2)
# model.load_state_dict(torch.load(os.path.join(run_directory, "model_params.pt")))
# model.eval()