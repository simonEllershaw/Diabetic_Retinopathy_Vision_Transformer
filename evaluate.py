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

def load_model(model_directory, model_name, device):
    model_fname = os.path.join(model_directory, "model_params.pt")
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
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
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.Softmax(1)(outputs)
        _, preds = torch.max(outputs, 1)
        
        prob_log[idx:min(idx+batch_size, num_samples)] = probs[:,1].detach()
        pred_log[idx:min(idx+batch_size, num_samples)] = preds.detach()
        idx += batch_size
    return prob_log, pred_log

def calc_metrics(labels, pred_log):
    metrics = {}
    metrics["conf_matrix"] = sklearn.metrics.confusion_matrix(labels, pred_log)
    metrics["accuracy"] = sklearn.metrics.accuracy_score(labels, pred_log)
    metrics["precision_score"] = sklearn.metrics.precision_score(labels, pred_log)
    metrics["recall_score"] = sklearn.metrics.recall_score(labels, pred_log)
    metrics["f1"] = sklearn.metrics.f1_score(labels, pred_log)  
    return metrics

def plot_precision_recall_curve(labels, prob_log, ax):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, prob_log)
    auc = sklearn.metrics.auc(recall, precision)
    pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax)
    no_skill = len(labels[labels==1]) / len(labels)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    return auc

def plot_ROC_curve(labels, prob_log, ax):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prob_log)
    auc = sklearn.metrics.auc(fpr, tpr)
    pr_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax)
    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return auc

def evaluate_model(model, device, dataloader, labels, batch_size, model_directory, phase):
    prob_log, pred_log = get_predictions(model, dataloader, len(labels), batch_size, device)
    metrics = calc_metrics(labels, pred_log)
    
    fig, ax = plt.subplots()
    metrics["Pre/Rec AUC"] = plot_precision_recall_curve(labels, prob_log, ax)
    plt.savefig(os.path.join(model_directory, f"precision_recall_curve{phase}.png"))
    
    fig, ax = plt.subplots()
    metrics["ROC AUC"] = plot_ROC_curve(labels, prob_log, ax)
    plt.savefig(os.path.join(model_directory, f"ROC_curve{phase}.png"))
    
    with open(os.path.join(model_directory, f"metrics{phase}.txt"), "w+") as f:
        f.write(pprint.pformat(metrics))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Load datasets split into train, val and test
    print(sys.argv)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "diabetic-retinopathy-detection"
    model_directory = sys.argv[2] if len(sys.argv) > 2 else "runs\\06_30_Grid_Search\\resnetv2_50x1_bitm_in21k\\0.01"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "resnet50"
    phase = sys.argv[4] if len(sys.argv) > 4 else "val"

    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, random_state=13)#, max_length=1000)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False, num_workers=4) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_directory, model_name, device)

    labels = datasets["val"].get_labels()
    evaluate_model(model, device, dataloader, labels, batch_size, model_directory, phase)