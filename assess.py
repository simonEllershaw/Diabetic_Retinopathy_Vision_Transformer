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

def load_model(model_directory, model_name, device):
    model_fname = os.path.join(model_directory, "model.pt")
    model_name = "vit_deit_small_patch16_224"
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    model.load_state_dict(torch.load("runs\\vit_deit_small_patch16_224_eyePACS_GrahamPreLRSearch\\0.1\model_params.pt"))
    model = model.eval().to(device)
    return model

def get_predictions(model, dataloader, num_samples, batch_size):
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

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Load datasets split into train, val and test
    data_directory = "diabetic-retinopathy-detection" 
    model_directory = "runs\\vit_deit_small_patch16_224_eyePACS_GrahamPreLRSearch\\0.01"
    model_name = "vit_deit_small_patch16_224"
    
    # data_directory = sys.argv[1]
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, random_state=13, max_length=1000)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_directory, model_name, device)

    labels = datasets["val"].get_labels()
    prob_log, pred_log = get_predictions(model, dataloader, len(datasets["val"]), batch_size)
    metrics = calc_metrics(labels, pred_log)
    fig, ax = plt.subplots()
    metrics["auc"] = plot_precision_recall_curve(labels, prob_log, ax)
    
    plt.savefig(os.path.join(model_directory, "precision_recall_curve.png"))
    with open(os.path.join(model_directory, "metrics.txt"), "w+") as f:
        f.write(pprint.pformat(metrics))
    