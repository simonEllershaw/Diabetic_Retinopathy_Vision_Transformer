import torch
import timm
import numpy as np
from datasets.eyePACS import EyePACS_Dataset
from datasets.messidor import Messidor_Dataset

import matplotlib.pyplot as plt
import sklearn.metrics
import os
import pprint
import sys
import time
import pandas as pd
from utilities.models import resize_ViT 
import json
from sklearn.metrics import confusion_matrix
import seaborn as sn
from shutil import copyfile 

def evaluate_model(model, device, model_directory, datasets, phase):
    dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=64, shuffle=False, num_workers=4) 
    prob_log = get_model_prob_outputs(model, dataloader, device).numpy()
    labels = datasets[phase].get_labels()
    dataset_name = f"{type(datasets[phase]).__name__}"
    evaluate_prob_outputs(prob_log, labels, model_directory, phase, dataset_name)

def evaluate_prob_outputs(prob_log, labels, model_directory, phase, dataset_name):
    fig, axs = plt.subplots(1, 2)
    metrics_log = {}

    auc, threshold = plot_precision_recall_curve(labels, prob_log, axs[0])
    metrics_log["threshold"] = threshold
    # Threshold is defined by validation set not test set!
    if phase != "val":
        threshold = get_threshold_from_val_metrics(model_directory)

    metrics_log["Pre/Rec AUC"] = auc
    metrics_log["ROC AUC"] = plot_ROC_curve(labels, prob_log, axs[1])

    pred_log = np.where(prob_log>=threshold, 1, 0)
    metrics_log = calc_metrics(labels, pred_log, metrics_log)
    metrics_log["prob_log"] = prob_log.tolist()
    metrics_log["pred_log"] = pred_log.tolist()
    metrics_log["false_positive_rate"] = calc_false_positive_rate(labels, pred_log)

    metrics_directory_phase = get_metrics_directory(model_directory, dataset_name, phase)
    os.makedirs(metrics_directory_phase, exist_ok=True) 

    axs[0].scatter(metrics_log["recall_score"], metrics_log["precision_score"], marker='x', color='black', label='Proposed Threshold')
    axs[1].scatter(metrics_log["false_positive_rate"], metrics_log["recall_score"], marker='x', color='black', label='Proposed Threshold')

    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    axs[0].legend()
    axs[1].legend()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(metrics_directory_phase, "AUC_curves.png"), dpi=100)

    with open(os.path.join(metrics_directory_phase, "metrics.txt"), "w+") as f:
        json.dump(metrics_log, f, indent=4)

def load_model(model_name, device, class_names, model_directory=None, model_resize=-1):
    model = timm.create_model(model_name, num_classes=len(class_names))
    if model_resize > 0:
        model = resize_ViT(model, model_resize)
    if model_directory is not None: 
        model_fname = os.path.join(model_directory, "model_params.pt")
        model.load_state_dict(torch.load(model_fname))
    model = model.eval().to(device)
    return model

def get_model_prob_outputs(model, dataloader, device):
    torch.set_grad_enabled(False)
    num_samples = len(dataloader.dataset)
    prob_log = torch.zeros(num_samples) 
    idx = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.nn.Softmax(1)(outputs)

        batch_size = len(inputs)
        prob_log[idx:idx+batch_size] = probs[:,1].detach()
        idx += batch_size
    return prob_log

def calc_metrics(labels, pred_log, metrics_log):
    metrics_log["accuracy"] = sklearn.metrics.accuracy_score(labels, pred_log)
    metrics_log["precision_score"] = sklearn.metrics.precision_score(labels, pred_log)
    metrics_log["recall_score"] = sklearn.metrics.recall_score(labels, pred_log)
    metrics_log["f1"] = sklearn.metrics.f1_score(labels, pred_log)
    metrics_log["conf_matrix"] = sklearn.metrics.confusion_matrix(labels, pred_log).tolist()
    return metrics_log

def plot_precision_recall_curve(labels, prob_log, ax, label=None):
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        no_skill = len(labels[labels==1]) / len(labels)
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, prob_log)
    opt_threshold = calc_shortest_distance_threshold(precision, recall, thresholds)
    auc = sklearn.metrics.auc(recall, precision)
    pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax, label=label)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return auc, opt_threshold

def calc_shortest_distance_threshold(precision, recall, thresholds):
    # distance from (1,1)
    distances = np.sqrt((1-precision)**2+(1-recall)**2)
    idx = np.argmin(distances)
    return np.float64(thresholds[idx])

def plot_ROC_curve(labels, prob_log, ax):
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prob_log)
    auc = sklearn.metrics.auc(fpr, tpr)
    pr_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return auc

def get_threshold_from_val_metrics(model_directory):
    val_metrics_file = os.path.join(get_metrics_directory(model_directory, "EyePACS_dataset", "val"), "metrics.txt")
    with open(val_metrics_file, "r") as f:
        val_metrics = json.load(f)
    return val_metrics["threshold"]

def plot_confusion_matrix(labels, predictions, ax, x_label, y_label):
    cm = confusion_matrix(labels, predictions)
    sn.heatmap(cm, annot=True, fmt='g', square=True, ax=ax, cbar=None, cmap="Blues")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def load_metrics(model_directory, dataset_name, phase="test"):
    metrics_dir = get_metrics_directory(model_directory, dataset_name, phase)
    with open(os.path.join(metrics_dir, "metrics.txt"), "r") as f:
        model_metrics = json.load(f)
    model_metrics = {key: np.array(value) for key, value in model_metrics.items()}
    return model_metrics

def get_metrics_directory(model_directory, dataset_name, phase):
    return os.path.join(model_directory, f"metrics_{dataset_name}_{phase}")

def generate_folders_of_disagreements(eval_dir, image_dir, labels, pred_ViT, pred_BiT, labels_df):
    # Create directories
    eval_dir = r"C:\Users\rmhisje\Documents\medical_ViT\eval_data"
    dir_ViT_correct = os.path.join(eval_dir, "ViT_correct")
    dir_BiT_correct = os.path.join(eval_dir, "BiT_correct")
    dir_both_wrong = os.path.join(eval_dir, "Both_wrong")

    if not os.path.exists(dir_ViT_correct):
        os.mkdir(dir_ViT_correct)
    if not os.path.exists(dir_BiT_correct):
        os.mkdir(dir_BiT_correct)
    if not os.path.exists(dir_both_wrong):
        os.mkdir(dir_both_wrong)

    # If models disagree put in relevant folder
    for idx, (label, ViT_predict, BiT_predict) in enumerate(zip(labels, pred_ViT, pred_BiT)):
        fname = labels_df.iloc[idx].image + ".jpeg"
        fpath = os.path.join(image_dir, fname)
        new_fname = f"{label}_{fname}"
        label_masked = 1 if label > 1 else 0
        if ViT_predict!=BiT_predict and ViT_predict==label_masked:
            copyfile(fpath, os.path.join(dir_ViT_correct, new_fname))
        elif ViT_predict!=BiT_predict and BiT_predict==label_masked:
            copyfile(fpath, os.path.join(dir_BiT_correct, new_fname))
        elif ViT_predict==BiT_predict and BiT_predict!=label_masked:
            copyfile(fpath, os.path.join(dir_both_wrong, new_fname))

def inter_model_matrix_comparision(labels, pred_ViT, pred_BiT, x_label, y_ticks):
    # Funky confusion matrix thing
    matrix = np.zeros((4, len(np.unique(labels))))
    for label, ViT_predict, BiT_predict in zip(labels, pred_ViT, pred_BiT):
        label_masked = 1 if label > 1 else 0
        if ViT_predict==BiT_predict and ViT_predict==label_masked:
            row = 0
        elif ViT_predict!=BiT_predict and ViT_predict==label_masked:
            row = 1
        elif ViT_predict!=BiT_predict and BiT_predict==label_masked:
            row = 2
        elif ViT_predict==BiT_predict and BiT_predict!=label_masked:
            row = 3
        else:
            print(label_masked, ViT_predict, BiT_predict)
        matrix[row, label] += 1
    matrix = matrix / matrix.sum(axis=0)
    sn.heatmap(matrix, annot=True, fmt=".2f", square=True, cbar=None, cmap="Blues")
    plt.xlabel(x_label)
    plt.yticks([0.5,1.5,2.5,3.5], y_ticks, rotation=0)
    plt.show()

def evaluate_ViT_BiT_ensemble_model(model_dir_ViT, model_dir_BiT, ensemble_dir, datasets):
    threshold_ViT = get_threshold_from_val_metrics(model_dir_ViT)
    threshold_BiT = get_threshold_from_val_metrics(model_dir_BiT)

    for phase in ["val", "test"]:
        metrics_ViT = load_metrics(model_dir_ViT, phase)
        metrics_BiT = load_metrics(model_dir_BiT, phase)
        labels = datasets[phase].get_labels()

        metrics_ViT["prob_log"] = metrics_ViT["prob_log"] * (0.5/threshold_ViT)
        metrics_BiT["prob_log"] = metrics_BiT["prob_log"] * (0.5/threshold_BiT)
        
        prob_log = (metrics_BiT["prob_log"] + metrics_ViT["prob_log"])/2
        evaluate_prob_outputs(prob_log, labels, ensemble_dir, phase)

def calc_false_positive_rate(labels, predictions):
    num_negatives = (len(labels)-np.sum(labels))
    num_false_positives = 0
    for predict, label in zip(predictions, labels):
        if label==0 and predict==1:
            num_false_positives += 1
    return num_false_positives/num_negatives

def plot_AUC_curves(labels, metrics):
    fig, ax = plt.subplots()
    for model_name, metrics in metrics.items():
        # Plot precision recall
        plot_precision_recall_curve(labels, metrics["prob_log"], ax, model_name)
        ax.scatter(metrics["recall_score"], metrics["precision_score"], marker='x', color='black')
        # Plot ROC
        # plot_ROC_curve(labels, metrics["prob_log"], axes[1])
        # axes[1].scatter(metrics["false_positive_rate"], metrics["recall_score"], marker='x', color='black')
    
    ax.set_box_aspect(1)
    # axes[1].set_box_aspect(1)
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.04,0.7), loc="upper left")
    plt.show()
    