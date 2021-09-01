# Standard libary imports
import os
import time
import argparse
# Deep learning imports
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
# from evaluation import evaluate
from utilities import visualisation, models
from datasets.eyePACS import EyePACS_Dataset
from datasets.eyePACS_masked import EyePACS_Masked_Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Entry into main training script')
    # Training variables
    parser.add_argument('-d', '--data_directory', type = str, default="data/eyePACs")
    parser.add_argument('-m', '--model', type = str, default="ResNet50", choices=["ResNet50", "ViT-S"])
    parser.add_argument('-t', '--pretraining', type = str, default="21k", choices=["21k", "DINO"])
    parser.add_argument('-l', '--lr', type = float, default=0.01)
    parser.add_argument('-n', '--num_steps', type = int, default=500)
    parser.add_argument('-w', '--num_warm_up_steps', type = int, default=100)
    parser.add_argument('-s', '--img_size', type = int, default=224)
    # Experiment variables
    parser.add_argument('-p', '--proportions', type = float, default=1)
    parser.add_argument('-g', '--grad_accum_off', action="store_true")
    parser.add_argument('-u', '--keep_ungradables', action="store_true")
    parser.add_argument('-a', '--no_data_aug_train', action="store_true")
    args = parser.parse_args()
    print(args)

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, use_inception_norm = models.load_model(args.model, args.pretraining, num_classes=2, img_size=args.img_size)
    model.to(device)   

    # Load datasets split into train, val and test
    dataset_names = ["train", "val", "test"]    
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    np.random.seed(13)
    full_dataset = EyePACS_Dataset(args.data_directory, img_size=args.img_size, random_state=13, use_inception_norm=use_inception_norm, remove_ungradables=not(args.keep_ungradables))
    # full_dataset = EyePACS_Masked_Dataset(data_directory, mask_size=4, img_size=args.img_size, use_inception_norm=use_inception_norm, random_state=13, remove_ungradables=not(args.keep_ungradables))
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    datasets["train"].augment=not(args.no_data_aug_train)
    if args.proportions < 1:
        datasets["train"].select_subset_of_data(0, int(len(datasets["train"])*args.proportions))
        datasets["val"].select_subset_of_data(0, int(len(datasets["val"])*args.proportions))

    # Grad accumulation
    batch_size = 512
    mini_batch_size= 16 if args.img_size==384 else 16
    if args.grad_accum_off:
        scaling = batch_size/mini_batch_size
        batch_size=mini_batch_size
        args.lr /= scaling
        args.num_steps *= scaling
        args.num_warm_up_steps *= scaling
    accumulation_steps = int(batch_size/mini_batch_size)
    
    # Setup dataloaders
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

    # LR schedule
    num_batches_per_train_epoch = len(datasets["train"]) / batch_size
    num_epochs = int(args.num_steps//num_batches_per_train_epoch)
    warmup_steps = int(args.num_warm_up_steps//num_batches_per_train_epoch)
    num_epochs_to_converge = 50

    # Optimisers
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = LRSchedules.WarmupCosineSchedule(optimizer, args.num_steps, args.num_warm_up_steps)
    grad_clip_norm = 1

    # Setup directory with name to unique identify training run and model
    dir_name = "-".join((args.model, args.pretraining, str(args.img_size), f"{type(full_dataset).__name__}", time.strftime("%m_%d_%H_%M_%S")))
    model_directory = os.path.join("runs", dir_name)
    os.makedirs(model_directory, exist_ok=True)
    print(model_directory)
    
    # Init tensorboard
    writer = SummaryWriter(model_directory)
    # Add input images to tensorboard for sanity check
    fig = visualisation.sample_batch(dataloaders["train"], class_names)
    writer.add_figure('Input/train', fig)
    fig = visualisation.sample_batch(dataloaders["val"], class_names)
    writer.add_figure('Input/val', fig)

    # Main training loop
    model, best_loss = train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs, device, dataset_sizes, len(class_names), writer, model_directory, warmup_steps, num_epochs_to_converge, accumulation_steps, grad_clip_norm)

    # Add sample inference outputs to tensorboard
    fig = visualisation.sample_batch(dataloaders["train"], class_names, model, device)
    writer.add_figure('Inference/train', fig)
    fig = visualisation.sample_batch(dataloaders["val"], class_names, model, device)
    writer.add_figure('Inference/val', fig)

    writer.close()

    evaluate.evaluate_model(model, device, model_directory, datasets, "val")
    evaluate.evaluate_model(model, device, model_directory, datasets, "test")