import os
import time

import timm
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import matplotlib.pyplot as plt

import LRSchedules
from train import train_model
import visualisation

# Data loading
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = 'hymenoptera_data'
datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}                    
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                        shuffle=True, num_workers=0)
                    for x in ['train', 'val']}   
class_names = datasets['train'].classes


# Set up directory to save model data
model_name = "resnet18"
dataset = "_hymenoptera_"
run_directory = os.path.join("runs", model_name+dataset+time.strftime("%m_%d_%H_%M_%S"))

# Set hyperparameters
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = timm.create_model("resnet18", pretrained=True, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = LRSchedules.WarmupCosineSchedule(optimizer, num_epochs)
grad_clip_norm = 1

# Init tensorboard
writer = SummaryWriter(run_directory)

# Add input images to tensorboard for sanity check
fig = visualisation.show_inputs(dataloaders["train"], class_names)
writer.add_figure('Input/train', fig)
fig = visualisation.show_inputs(dataloaders["val"], class_names)
writer.add_figure('Input/val', fig)

# Main training loop
model, best_acc = train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, writer, run_directory, grad_clip_norm)

# Add sample inference outputs to tensorboard
fig = visualisation.show_inference(dataloaders["train"], class_names, model, device)
writer.add_figure('Inference/train', fig)
fig = visualisation.show_inference(dataloaders["val"], class_names, model, device)
writer.add_figure('Inference/val', fig)

writer.close()

model = timm.create_model("resnet18", pretrained=True, num_classes=2)
model.load_state_dict(torch.load(os.path.join(run_directory, "model_params.pt")))
model.eval()