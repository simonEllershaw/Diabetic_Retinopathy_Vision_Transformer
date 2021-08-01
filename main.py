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
from vision_transformer_utils import resize_ViT
# from colouredSquares import Coloured_Squares_Dataset
from eyePACS_masked import EyePACS_Masked_Dataset
from messidor import Messidor_Dataset
# Testing
import numpy as np
from torchvision import transforms, datasets
import metrics
import sklearn.metrics

if __name__ == "__main__":
    # Set up directory for experiment
    print(sys.argv, len(sys.argv))
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "data/eyePACs"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "vit_small_patch16_224_in21k"
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    num_warm_up_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    img_size = int(sys.argv[6]) if len(sys.argv) > 6 else 224
    resize_model = True if (len(sys.argv) > 7 and int(sys.argv[7]) > 0) else False
    proportions = float(sys.argv[8]) if (len(sys.argv) > 8) else 1
    # mini_batching_turn_off = True if (len(sys.argv) > 8 and int(sys.argv[8]) > 0) else False
    # remove_ungradables = False if (len(sys.argv) > 9 and int(sys.argv[9]) > 0) else True
    data_aug_train = False if (len(sys.argv) > 10 and int(sys.argv[10]) > 0) else True

    # Load datasets split into train, val and test
    dataset_names = ["train", "val", "test"]    
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    np.random.seed(13)
    full_dataset = EyePACS_Dataset(data_directory, img_size=img_size, random_state=13)
    # full_dataset =  Messidor_Dataset(data_directory, random_state=13, img_size=img_size)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    datasets["train"].augment=data_aug_train
    if proportions < 1:
        datasets["train"].select_subset_of_data(0, int(len(datasets["train"])*proportions))
        datasets["val"].select_subset_of_data(0, int(len(datasets["val"])*proportions))

    # Setup dataloaders
    batch_size = 512
    mini_batch_size= 16 if img_size==384 else 64

    if mini_batching_turn_off:
        print("mini-batching")
        scaling = batch_size/mini_batch_size
        batch_size=mini_batch_size
        lr /= scaling
        num_steps *= scaling
        num_warm_up_steps *= scaling

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
    
    if "dino" in model_name:
        model = torch.hub.load('facebookresearch/dino:main', model_name)    
        if "resnet" in model_name:
            model = torch.nn.Sequential(model, torch.nn.Linear(2048, 2))
        else:
            model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names)).to(device)
        if resize_model:
            print("model resize")
            model = resize_ViT(model, img_size)
    model = model.to(device)    

    num_batches_per_train_epoch = len(datasets["train"]) / batch_size
    num_epochs = int(num_steps//num_batches_per_train_epoch)
    print(num_epochs)
    warmup_steps = int(num_warm_up_steps//num_batches_per_train_epoch)
    num_epochs_to_converge = 50
    grad_clip_norm = 1

    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = LRSchedules.WarmupCosineSchedule(optimizer, num_steps, num_warm_up_steps)

    dataset_name = f"_{type(full_dataset).__name__}_"
    model_directory = os.path.join("runs", model_name + dataset_name + time.strftime("%m_%d_%H_%M_%S"))
    os.mkdir(model_directory)
    print(model_directory)
    
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

    evaluate.evaluate_model(model, device, model_directory, datasets, "val")
    evaluate.evaluate_model(model, device, model_directory, datasets, "test")