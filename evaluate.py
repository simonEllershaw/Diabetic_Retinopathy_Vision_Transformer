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

def evaluate_models(models, device, dataloader, labels, model_directory, phase, model_name):
    fig, axs = plt.subplots(1, 2)
    metrics_log = {}
    
    prob_log = get_model_prob_outputs(model, dataloader, device)

    auc, threshold = plot_precision_recall_curve(labels, prob_log, axs[0], model_name)
    metrics_log["threshold"] = threshold
    metrics_log["Pre/Rec AUC"] = auc
    metrics_log["ROC AUC"] = plot_ROC_curve(labels, prob_log, axs[1], model_name)
    
    # Threshold is defined by validation set not test set!
    if phase != "val":
        threshold = get_threshold_from_val_metrics(model_directory)

    pred_log = torch.where(prob_log>=threshold, 1, 0)
    metrics_log = calc_metrics(labels, pred_log, metrics_log)
    metrics_log["prob_log"] = prob_log.numpy().tolist()
    metrics_log["pred_log"] = prob_log.numpy().tolist()

    metrics_directory = os.path.join(model_directory, f"metrics_{phase}")
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

def plot_precision_recall_curve(labels, prob_log, ax, model_name):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, prob_log)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    threshold = np.float64(thresholds[ix])
    auc = sklearn.metrics.auc(recall, precision)
    pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax, label=model_name)
    ax.scatter(recall[ix], precision[ix], marker='o', color='black', label=f'Best {model_name}')
    
    no_skill = len(labels[labels==1]) / len(labels)
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    return auc, threshold

def plot_ROC_curve(labels, prob_log, ax, model_name):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prob_log)
    auc = sklearn.metrics.auc(fpr, tpr)
    pr_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax, label=model_name)
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

if __name__ == "__main__":
    # Load datasets split into train, val and test
    print(sys.argv)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "diabetic-retinopathy-detection"
    model_directory = sys.argv[2] if len(sys.argv) > 2 else r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "vit_small_patch16_224_in21k"#"resnetv2_50x1_bitm_in21k"
    phase = sys.argv[4] if len(sys.argv) > 4 else "val"

    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, random_state=13, img_size=384)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    batch_size = 64
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_directory, model_name, device, class_names, 384)
    phase = "val"
    dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False, num_workers=4) 
    labels = datasets[phase].get_labels()
    evaluate_models(model, device, dataloader, labels, model_directory, phase, model_name)
    phase = "test"
    dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False, num_workers=4) 
    labels = datasets[phase].get_labels()
    evaluate_models(model, device, dataloader, labels, model_directory, "test", model_name)


    # model_directory = r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01"
    # with open(os.path.join(model_directory, f"outputs"), "r") as f:
    #     model_outputs_ViT = json.load(f)
    # model_outputs_ViT = {key: np.array(value) for key, value in model_outputs_ViT.items()}
    
    # model_directory = r"runs\384_Run_Baseline\resnetv2_50x1_bitm_in21k_eyePACS_LR_0.01"
    # with open(os.path.join(model_directory, f"outputs"), "r") as f:
    #     model_outputs_BiT = json.load(f)
    # model_outputs_BiT = {key: np.array(value) for key, value in model_outputs_BiT.items()}

    # from sklearn.metrics import confusion_matrix
    # print(sum(model_outputs_ViT["pred_log"]))
    # print(sum(model_outputs_BiT["pred_log"]))
    # print(confusion_matrix(labels, model_outputs_BiT["pred_log"]))
    # print(confusion_matrix(labels, model_outputs_ViT["pred_log"]))
    # print(confusion_matrix(model_outputs_ViT["pred_log"], model_outputs_BiT["pred_log"]))
    