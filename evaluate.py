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

def load_model(model_directory, model_name, device, class_names):
    model_fname = os.path.join(model_directory, "model_params.pt")
    model = timm.create_model(model_name, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_fname))
    model = model.eval().to(device)
    return model

def get_predictions(model, dataloader, num_samples, batch_size, device):
    torch.set_grad_enabled(False)
    prob_log = torch.zeros(num_samples) 
    pred_log = torch.zeros(num_samples) 
    idx = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.nn.Softmax(1)(outputs)
        _, preds = torch.max(outputs, 1)
        batch_size = len(inputs)
        prob_log[idx:idx+batch_size] = probs[:,1].detach()
        pred_log[idx:idx+batch_size] = preds.detach()
        idx += batch_size
    return prob_log, pred_log

def init_metrics_log_dict():
    keys = ["accuracy", "precision_score", "recall_score", "f1", "Pre/Rec AUC", "ROC AUC", "conf_matrix"]
    metrics_log = {key: [] for key in keys}
    return metrics_log

def calc_metrics(labels, pred_log, metrics_log):
    metrics_log["conf_matrix"].append(sklearn.metrics.confusion_matrix(labels, pred_log))
    metrics_log["accuracy"].append(sklearn.metrics.accuracy_score(labels, pred_log))
    metrics_log["precision_score"].append(sklearn.metrics.precision_score(labels, pred_log))
    metrics_log["recall_score"].append(sklearn.metrics.recall_score(labels, pred_log))
    metrics_log["f1"].append(sklearn.metrics.f1_score(labels, pred_log))
    return metrics_log

def plot_precision_recall_curve(labels, prob_log, ax, model_name):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, prob_log)
    auc = sklearn.metrics.auc(recall, precision)
    pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax, label=model_name)
    no_skill = len(labels[labels==1]) / len(labels)
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    return auc

def plot_ROC_curve(labels, prob_log, ax, model_name):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prob_log)
    auc = sklearn.metrics.auc(fpr, tpr)
    pr_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax, label=model_name)
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return auc

def evaluate_models(models, device, dataloader, labels, batch_size, eval_directory, phase, model_names):
    directory = os.path.join(eval_directory, time.strftime('%m_%d_%H_%M_%S'))
    os.mkdir(directory)
    fig, axs = plt.subplots(1, 2)
    metrics_log = init_metrics_log_dict()
    for model, model_name in zip(models, model_names):
        model.eval()
        prob_log, pred_log = get_predictions(model, dataloader, len(labels), batch_size, device)
        
        metrics_log = calc_metrics(labels, pred_log, metrics_log)
        metrics_log["Pre/Rec AUC"].append(plot_precision_recall_curve(labels, prob_log, axs[0], model_name))
        metrics_log["ROC AUC"].append(plot_ROC_curve(labels, prob_log, axs[1], model_name))
    
    axs[0].legend()
    axs[1].legend()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(directory, f"eval_curves_{phase}.png"), dpi=100)

    metrics_log_df = pd.DataFrame.from_dict(metrics_log, orient='index', columns=model_names)
    metrics_log_df.to_csv(os.path.join(directory, f"metrics_{phase}.txt"))

if __name__ == "__main__":
    # Load datasets split into train, val and test
    print(sys.argv)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "diabetic-retinopathy-detection"
    model_directorys = sys.argv[2] if len(sys.argv) > 2 else ["runs\\06_30_Grid_Search\\resnetv2_50x1_bitm_in21k\\0.01", "runs\\06_30_Grid_Search\\vit_small_patch16_224_in21k\\0.01"]
    model_names = sys.argv[3] if len(sys.argv) > 3 else ["resnetv2_50x1_bitm_in21k", "vit_small_patch16_224_in21k"]
    phase = sys.argv[4] if len(sys.argv) > 4 else "test"
    eval_directory = "eval_data"

    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, random_state=13)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False, num_workers=4) 
    labels = datasets[phase].get_labels()
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    for model_name, model_directory in zip(model_names, model_directories):
        models.append(load_model(model_directory, model_name, device, class_names))

    evaluate_models(models, device, dataloader, labels, batch_size, eval_directory, phase, model_names)