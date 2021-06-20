import metrics
import visualisation
import torch
import timm
import numpy as np
from eyePACS import EyePACS_Dataset
import matplotlib.pyplot as plt
import sklearn.metrics

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Load datasets split into train, val and test
    dataset_names = ["train", "val", "test"]
    data_directory = "diabetic-retinopathy-detection" 
    # data_directory = sys.argv[1]
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    full_dataset = EyePACS_Dataset(data_directory, max_length=1000, random_state=13)
    class_names = full_dataset.class_names

    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    # datasets["train"].augment=True
    # Setup dataloaders
    batch_size = 100
    num_workers = 4
    dataset_sizes = {name: len(datasets[name]) for name in dataset_names}                  
    dataloaders = {name: torch.utils.data.DataLoader(datasets[name], batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
                        for name in dataset_names}   
    dataloaders["train"].shuffle = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "vit_deit_small_patch16_224"
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    model.load_state_dict(torch.load("runs\\vit_deit_small_patch16_224_eyePACS_06_20_14_21_42\\model_params.pt"))
    model = model.train().to(device)

    confusion_matrix = torch.zeros(len(class_names), len(class_names))
    prob_log = torch.zeros(len(datasets["val"])) 
    pred_log = torch.zeros(len(datasets["val"])) 
    idx = 0
    torch.set_grad_enabled(False)

    for inputs, labels, _ in dataloaders["val"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.Softmax(1)(outputs)
        _, preds = torch.max(outputs, 1)
        prob_log[idx:min(idx+batch_size, len(datasets["val"]))] = probs[:,1].detach()
        pred_log[idx:min(idx+batch_size, len(datasets["val"]))] = preds.detach()
        idx += batch_size
        confusion_matrix = metrics.update_conf_matrix(confusion_matrix, labels, preds)
    
    labels = datasets["val"].get_labels()
    print(confusion_matrix)
    print(sklearn.metrics.confusion_matrix(labels, pred_log))
    print(metrics.calc_binary_f1_score(confusion_matrix))
    f1 = sklearn.metrics.f1_score(labels, pred_log)
    print(f1)
    # precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, prob_log)
    # auc = sklearn.metrics.auc(recall, precision)
    # pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    # no_skill = len(labels[labels==1]) / len(labels)
    # plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plt.show()
    # print(auc, f1)