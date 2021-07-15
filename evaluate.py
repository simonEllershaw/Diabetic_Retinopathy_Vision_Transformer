import metrics
import visualisation
import torch
import timm
import numpy as np
from eyePACS import EyePACS_Dataset
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import pprint
import sys
import time
import pandas as pd
from vision_transformer_utils import resize_ViT 
import json
from sklearn.metrics import confusion_matrix
import seaborn as sn
from shutil import copyfile


def evaluate_models(models, device, dataloader, labels, model_directory, phase):
    fig, axs = plt.subplots(1, 2)
    metrics_log = {}
    
    prob_log = get_model_prob_outputs(model, dataloader, device)

    auc, threshold = plot_precision_recall_curve(labels, prob_log, axs[0])
    metrics_log["threshold"] = threshold
    # Threshold is defined by validation set not test set!
    if phase != "val":
        threshold = get_threshold_from_val_metrics(model_directory)

    metrics_log["Pre/Rec AUC"] = auc
    metrics_log["ROC AUC"] = plot_ROC_curve(labels, prob_log, axs[1])


    pred_log = torch.where(prob_log>=threshold, 1, 0)
    metrics_log = calc_metrics(labels, pred_log, metrics_log)
    metrics_log["prob_log"] = prob_log.numpy().tolist()
    metrics_log["pred_log"] = pred_log.numpy().tolist()

    axs[0].scatter(metrics_log["recall_score"], metrics_log["precision_score"], marker='o', color='black', label='Proposed Threshold')

    metrics_directory = os.path.join(model_directory, f"metrics_{phase}")
    if not os.path.exists(metrics_directory):
        os.mkdir(metrics_directory) 

    axs[0].legend()
    axs[1].legend()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(metrics_directory, "AUC_curves.png"), dpi=100)

    with open(os.path.join(metrics_directory, "metrics.txt"), "w+") as f:
        json.dump(metrics_log, f, indent=4)

def load_model(model_directory, model_name, device, class_names, model_resize=-1):
    model_fname = os.path.join(model_directory, "model_params.pt")
    model = timm.create_model(model_name, num_classes=len(class_names))
    if model_resize > 0:
        model = resize_ViT(model, model_resize)
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

def plot_precision_recall_curve(labels, prob_log, ax):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, prob_log)
    opt_threshold = calc_shortest_distance_threshold(precision, recall, thresholds)
    auc = sklearn.metrics.auc(recall, precision)
    pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax)
    
    no_skill = len(labels[labels==1]) / len(labels)
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    return auc, opt_threshold

def calc_shortest_distance_threshold(precision, recall, thresholds):
    # distance from (1,1)
    distances = np.sqrt((1-precision)**2+(1-recall)**2)
    idx = np.argmin(distances)
    return np.float64(thresholds[idx])

def plot_ROC_curve(labels, prob_log, ax):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prob_log)
    auc = sklearn.metrics.auc(fpr, tpr)
    pr_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax)
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return auc

def get_threshold_from_val_metrics(model_directory):
    val_metrics_file = os.path.join(model_directory, "metrics_val", "metrics.txt")
    with open(val_metrics_file, "r") as f:
        val_metrics = json.load(f)
    return val_metrics["threshold"]

def plot_confusion_matrix(labels, predictions, ax, x_label, y_label):
    cm = confusion_matrix(labels, predictions)
    sn.heatmap(cm, annot=True, fmt='g', square=True, ax=ax, cbar=None, cmap="Blues")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def load_test_metrics(model_directory):
    with open(os.path.join(model_directory, "metrics_test", "metrics.txt"), "r") as f:
        model_metrics = json.load(f)
    model_metrics = {key: np.array(value) for key, value in model_metrics.items()}
    return model_metrics

def inter_model_matrix_comparision(eval_dir, image_dir, labels, pred_ViT, pred_BiT, labels_df):
    # Funky confusion matrix thing
    eval_dir = r"C:\Users\rmhisje\Documents\medical_ViT\eval_data"
    if not os.path.exists(os.path.join(eval_dir, "ViT_correct")):
        os.mkdir(os.path.join(eval_dir, "ViT_correct"))
    if not os.path.exists(os.path.join(eval_dir, "BiT_correct")):
        os.mkdir(os.path.join(eval_dir, "BiT_correct"))
    if not os.path.exists(os.path.join(eval_dir, "Both_wrong")):
        os.mkdir(os.path.join(eval_dir, "Both_wrong"))

    matrix = np.zeros((4, len(np.unique(labels))))
    for idx, (label, ViT_predict, BiT_predict) in enumerate(zip(labels, pred_ViT, pred_BiT)):
        fname = labels_df.iloc[idx].image + ".jpeg"
        fpath = os.path.join(image_dir, fname)
        label_masked = 1 if label > 1 else 0
        if ViT_predict==BiT_predict and ViT_predict==label_masked:
            row = 0
        elif ViT_predict!=BiT_predict and ViT_predict==label_masked:
            row = 1
            copyfile(fpath, os.path.join(eval_dir, "ViT_correct", f"{label}_{fname}"))
        elif ViT_predict!=BiT_predict and BiT_predict==label_masked:
            row = 2
            copyfile(fpath, os.path.join(eval_dir, "BiT_correct", f"{label}_{fname}"))
        elif ViT_predict==BiT_predict and BiT_predict!=label_masked:
            row = 3
            copyfile(fpath, os.path.join(eval_dir, "Both_wrong", f"{label}_{fname}"))
        else:
            print(label_masked, ViT_predict, BiT_predict)
        matrix[row, label] += 1
    matrix = matrix / matrix.sum(axis=0)
    sn.heatmap(matrix, annot=True, fmt=".2f", square=True, cbar=None, cmap="Blues")
    plt.xlabel("Label")
    plt.yticks([0.5,1.5,2.5,3.5],["Both Correct","Only ViT Correct","Only BiT Correct","Both Wrong"], rotation=0)
    plt.show()


if __name__ == "__main__":
    # Load datasets split into train, val and test
    print(sys.argv)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "diabetic-retinopathy-detection"
    model_directory = sys.argv[2] if len(sys.argv) > 2 else r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "vit_small_patch16_224_in21k"#"resnetv2_50x1_bitm_in21k" vit_small_patch16_224_in21k
    phase = sys.argv[4] if len(sys.argv) > 4 else "val"

    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, random_state=13, img_size=384)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    batch_size = 64
    
    # Save metrics for model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = load_model(model_directory, model_name, device, class_names, 384)
    # phase = "val"
    # dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False, num_workers=4) 
    # labels = datasets[phase].get_labels()
    # evaluate_models(model, device, dataloader, labels, model_directory, phase)
    phase = "test"
    # dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False, num_workers=4) 
    labels = datasets[phase].get_labels()
    # evaluate_models(model, device, dataloader, labels, model_directory, "test")

    # Load metrics from model
    labels = datasets["test"].get_labels()
    metrics_ViT = load_test_metrics(r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01")
    metrics_BiT = load_test_metrics(r"runs\384_Run_Baseline\resnetv2_50x1_bitm_in21k_eyePACS_LR_0.01")
    model_directory = r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01"

    # Plot confusion matrices
    # fig, axes = plt.subplots(1, 3)
    # plot_confusion_matrix(labels, metrics_BiT["pred_log"], axes[0], "BiT prediction", "eyePACs label")
    # plot_confusion_matrix(labels, metrics_ViT["pred_log"], axes[1], "ViT prediction", "eyePACs label")
    # plot_confusion_matrix(metrics_ViT["pred_log"], metrics_BiT["pred_log"], axes[2], "ViT prediction", "BiT prediction")
    # plt.show()

    # Funky comparison matrix thing
    eval_dir = r"C:\Users\rmhisje\Documents\medical_ViT\eval_data"
    image_dir = r"C:\Users\rmhisje\Documents\medical_ViT\diabetic-retinopathy-detection\preprocessed_448"
    inter_model_matrix_comparision(eval_dir, image_dir, labels, metrics_ViT["pred_log"], metrics_BiT["pred_log"], datasets["test"].labels_df)