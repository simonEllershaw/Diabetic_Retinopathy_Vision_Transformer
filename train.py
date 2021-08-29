# Standard libary imports
import os
import time
import sys
import argparse
# Deep learning imports
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
import sklearn.utils
import numpy as np
# Internal imports
from training import LRSchedules
from training.train import train_model
from evaluation import evaluate
from utils.vision_transformer_utils import resize_ViT
from utils import visualisation
from datasets.eyePACS import EyePACS_Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Entry into main training script')
    # Training variables
    parser.add_argument('-d', '--data_directory', type = str, nargs = 1, default="data/eyePACs")
    parser.add_argument('-m', '--model_name', type = str, nargs = 1, default="resnetv2_50x1_bitm_in21k", choices=["vit_small_patch16_224_in21k", "resnetv2_50x1_bitm_in21k", "dino_resnet50", "dino_vits16"])
    parser.add_argument('-l', '--lr', type = float, nargs = 1, default=0.01)
    parser.add_argument('-n', '--num_steps', type = int, nargs = 1, default=500)
    parser.add_argument('-w', '--num_warm_up_steps', type = int, nargs = 1, default=100)
    parser.add_argument('-s', '--img_size', type = int, nargs = 1, default=224)
    parser.add_argument('-r', '--resize_model', action="store_true", default=False)
    # Experiment variables
    parser.add_argument('-p', '--proportions', type = float, nargs = 1, default=0.1)
    parser.add_argument('-i', '--use_inception_norm', action="store_true", default=True)
    parser.add_argument('-g', '--grad_accum', action="store_true", default=True)
    parser.add_argument('-u', '--remove_ungradables', action="store_true", default=True)
    parser.add_argument('-a', '--data_aug_train', action="store_true", default=True)
    args = parser.parse_args()
    print(args)
    
    # Load datasets split into train, val and test
    dataset_names = ["train", "val", "test"]    
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    np.random.seed(13)
    full_dataset = EyePACS_Dataset(args.data_directory, img_size=args.img_size, random_state=13, use_inception_norm=args.use_inception_norm)
    # full_dataset =  Messidor_Dataset(data_directory, random_state=13, img_size=img_size)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    datasets["train"].augment=args.data_aug_train
    if args.proportions < 1:
        datasets["train"].select_subset_of_data(0, int(len(datasets["train"])*args.proportions))
        datasets["val"].select_subset_of_data(0, int(len(datasets["val"])*args.proportions))

    # Setup dataloaders
    batch_size = 512
    mini_batch_size= 16 if args.img_size==384 else 16

    if not args.grad_accum:
        scaling = batch_size/mini_batch_size
        batch_size=mini_batch_size
        args.lr /= scaling
        args.num_steps *= scaling
        args.num_warm_up_steps *= scaling

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
    
    if "dino" in args.model_name:
        model = torch.hub.load('facebookresearch/dino:main', args.model_name)    
        if "resnet" in args.model_name:
            model = torch.nn.Sequential(model, torch.nn.Linear(2048, 2))
        else:
            model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
    else:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=len(class_names)).to(device)
        if args.resize_model:
            print("model resize")
            model = resize_ViT(model, args.img_size)
    model = model.to(device)    

    num_batches_per_train_epoch = len(datasets["train"]) / batch_size
    num_epochs = 1#int(args.num_steps//num_batches_per_train_epoch)
    print(num_epochs)
    warmup_steps = int(args.num_warm_up_steps//num_batches_per_train_epoch)
    num_epochs_to_converge = 50
    grad_clip_norm = 1

    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = LRSchedules.WarmupCosineSchedule(optimizer, args.num_steps, args.num_warm_up_steps)

    dataset_name = f"_{type(full_dataset).__name__}_"
    model_directory = os.path.join("runs", args.model_name + dataset_name + time.strftime("%m_%d_%H_%M_%S"))
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