from vit_explain.vit_rollout import VITAttentionRollout
from vit_explain.vit_grad_rollout import VITAttentionGradRollout
import evaluate
from eyePACS_masked import EyePACS_Masked_Dataset
from IDRiD import IDRiD_Dataset
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import visualisation
import math
from pytorch_grad_cam import GradCAM
import cv2
import sklearn.metrics
import pandas as pd

class Last_Layer:
    def __init__(self, model, attention_layer_name='attn_drop'):
        self.model = model
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []
    

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return map_attention_of_class_token_last_layer(self.attentions[-1])

def map_attention_of_class_token_last_layer(last_layer_attentions):
    class_attentions = last_layer_attentions[0, :, 0, 1:]
    width = int(class_attentions.size(-1)**0.5)
    class_attentions = class_attentions.reshape(-1, width, width).numpy()
    class_attentions = class_attentions.sum(0)
    return class_attentions

def get_random_map(image_batch):
    num_patches = image_batch.size(-1)//16
    return np.random.rand(num_patches, num_patches)

def generate_heatmaps(sample, interpreter):
    image, _, seg_cum, _ = sample
    heatmap = interpreter(image.unsqueeze(0).to(device))
    if not (heatmap.shape == seg_cum.shape):
            heatmap = get_patch_heatmap(heatmap[0])
    heatmap = heatmap/heatmap.sum()
    intersect_map = calc_intersect_map(seg_cum, heatmap)
    return heatmap, intersect_map

def threshold_heatmap(heatmap, mass_threshold=0.6):
    if mass_threshold < 1:
        heatmap_sorted = np.sort(heatmap.ravel())
        total_mass = heatmap.sum()
        heatmap_cum_sum = heatmap_sorted.cumsum()
        greater_than_mass_threshold = heatmap_cum_sum > total_mass*(1-mass_threshold)
        threshold = heatmap_sorted[greater_than_mass_threshold.searchsorted(True)]
        heatmap = np.where(heatmap>=threshold, heatmap, 0)
    heatmap /= heatmap.sum()
    return heatmap

def calc_intersect_map(seg_cum, heatmap):
    intersect_map = np.multiply(seg_cum, heatmap)
    return intersect_map

def get_patch_heatmap(heatmap, patch_size=16):
    num_patches = heatmap.shape[-1] // patch_size
    return heatmap.reshape(num_patches, patch_size, num_patches, patch_size).sum(axis=(1, 3))

def get_hit_rate(dataset, interpreter):
    num_hits = 0
    for idx in range(len(dataset)):
        _, _, seg_cum, _ = dataset[idx]
        heatmap, _ = generate_heatmaps(dataset[idx], interpreter)
        top_index = np.argmax(heatmap.flatten())
        if seg_cum.flatten()[top_index] == 1:
            num_hits += 1
    return num_hits/len(dataset) #

def calc_weighted_sensitivity(dataset, interpreter):
    sensitivity = 0
    for idx in range(len(dataset)):
        _, intersect_map = generate_heatmaps(dataset[idx], interpreter)
        sensitivity += intersect_map.sum()
    return sensitivity/len(dataset)

def output_visualisation_column(dataset, interpreter, interpreter_label, axes, num_samples):
    for idx in range(num_samples):
        image, _, _, _ = dataset[idx]
        heatmap, _ = generate_heatmaps(dataset[idx], interpreter)
        visualisation.imshow(image, ax=axes[idx])
        axes[idx].imshow(cv2.resize(heatmap, dsize=(image.size(-1), image.size(-1)), interpolation=cv2.INTER_NEAREST), alpha=0.5)#, vmin=0, vmax=0.02)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    axes[0].title.set_text(interpreter_label)
    
def calc_intersection_by_type(dataset, interpreter):
    intersect_by_type = np.zeros(4)
    for idx in range(len(dataset)):
        _, heatmap_threshold, intersect_map = generate_heatmaps(dataset[idx], interpreter)
        intersect_by_type[:-1] += np.count_nonzero(intersect_map, (0,1))
        intersect_by_type[-1] += np.count_nonzero(heatmap_threshold) - np.count_nonzero(intersect_map)
    return intersect_by_type/intersect_by_type.sum()

def get_labels_and_pred_array(dataset, interpreter):
    sample_seg_cum = dataset[0][2]
    num_patches_per_image = sample_seg_cum.shape[-1]**2
    y_true = np.zeros(num_patches_per_image*len(dataset))
    y_pred = np.zeros(num_patches_per_image*len(dataset))

    for idx in range(len(dataset)):
        _, _, seg_cum, _ = dataset[idx]
        heatmap, _ = generate_heatmaps(dataset[idx], interpreter)
        y_true[idx*num_patches_per_image:(idx+1)*num_patches_per_image] = seg_cum.flatten()
        y_pred[idx*num_patches_per_image:(idx+1)*num_patches_per_image] = heatmap.flatten()
    
    return y_true, y_pred

def calc_pre_rec_curve(y_true, y_pred):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    auc = sklearn.metrics.auc(recall, precision)
    return precision, recall, thresholds, auc

def plot_precision_recall_curve(precision, recall, y_true, ax, label):
    if "No Skill" not in ax.get_legend_handles_labels()[1]:
        no_skill = len(y_true[y_true==1]) / len(y_true)
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    
    pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax, label=label)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

if __name__ == "__main__":
    model_dir_ViT = r"runs\384_Strong_Aug\vit_small_patch16_224_in21k\LR_0.01"
    model_dir_BiT = r"runs\384_Strong_Aug\resnetv2_50x1_bitm_in21k\LR_0.01"

    np.random.seed(13)
    dataset = IDRiD_Dataset(r'data\idrid', img_size=384)
    class_names = ["Healthy", "Refer"]
    # dataset_names = ["train", "test"]    
    # dataset_proportions = np.array([0.5, 0.5])
    # datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    experiment = "AUC"

    if experiment == "intersect bar chart":
        hit_rates = []
        sensitivities = []
        methods = []
    elif experiment == "side by side vis":
        num_samples = 5
        fig, axes = plt.subplots(num_samples, 8, squeeze=False)
        col_num = 2
    elif experiment == "feature detected":
        feature_proportions = {}
    elif experiment == "AUC":
        fig, ax = plt.subplots()
        AUC = {}

    
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')    
    model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
    model.load_state_dict(torch.load(r"runs\dino\vits_16\384\LR0.01\model_params.pt"))
    model = model.to(device)

    interpreter = Last_Layer(model[0])
    interpreter_label = "Last Layer\n(ViT-S-DINO)"

    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        output_visualisation_column(dataset, interpreter, interpreter_label, axes[:,col_num], num_samples)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter)
    elif experiment == "AUC":
        y_true, y_pred = get_labels_and_pred_array(dataset, interpreter)
        precision, recall, thresholds, auc = calc_pre_rec_curve(y_true, y_pred)
        AUC[interpreter_label] = auc
        plot_precision_recall_curve(precision, recall, y_true, ax, interpreter_label)

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')    
    model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
    model.load_state_dict(torch.load(r"runs\dino\vits_16\384\LR0.01\model_params.pt"))
    model = model.to(device)

    interpreter = VITAttentionRollout(model[0], discard_ratio=0.9, head_fusion="max")
    interpreter_label = "Attention Rollout\n(ViT-S-DINO)"
    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        output_visualisation_column(dataset, interpreter, interpreter_label, axes[:,col_num], num_samples)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter)
    elif experiment == "AUC":
        y_true, y_pred = get_labels_and_pred_array(dataset, interpreter)
        precision, recall, thresholds, auc = calc_pre_rec_curve(y_true, y_pred)
        AUC[interpreter_label] = auc
        plot_precision_recall_curve(precision, recall, y_true, ax, interpreter_label)

    model = evaluate.load_model("vit_small_patch16_224_in21k", device, class_names, model_dir_ViT, 384)  
    interpreter = Last_Layer(model)
    interpreter_label = "Last Layer\n(ViT-S-21k)"
    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        output_visualisation_column(dataset, interpreter, interpreter_label, axes[:,col_num], num_samples)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter)
    elif experiment == "AUC":
        y_true, y_pred = get_labels_and_pred_array(dataset, interpreter)
        precision, recall, thresholds, auc = calc_pre_rec_curve(y_true, y_pred)
        AUC[interpreter_label] = auc
        plot_precision_recall_curve(precision, recall, y_true, ax, interpreter_label)
    
    model = evaluate.load_model("vit_small_patch16_224_in21k", device, class_names, model_dir_ViT, 384)  
    interpreter = VITAttentionRollout(model, discard_ratio=0.9, head_fusion="max")
    interpreter_label = "Attention Rollout\n(ViT-S-21k)"
    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        output_visualisation_column(dataset, interpreter, interpreter_label, axes[:,col_num], num_samples)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter)
    elif experiment == "AUC":
        y_true, y_pred = get_labels_and_pred_array(dataset, interpreter)
        precision, recall, thresholds, auc = calc_pre_rec_curve(y_true, y_pred)
        AUC[interpreter_label] = auc
        plot_precision_recall_curve(precision, recall, y_true, ax, interpreter_label)
    
    model = evaluate.load_model("resnetv2_50x1_bitm_in21k", device, class_names, model_dir_BiT)   
    interpreter = GradCAM(model=model, target_layer=model.stages[-1], use_cuda=True)
    interpreter_label = "GradCAM\n(ResNet50-21k)"
    mass_threshold = 1
    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        output_visualisation_column(dataset, interpreter, interpreter_label, axes[:,col_num], num_samples)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter)
    elif experiment == "AUC":
        y_true, y_pred = get_labels_and_pred_array(dataset, interpreter)
        precision, recall, thresholds, auc = calc_pre_rec_curve(y_true, y_pred)
        AUC[interpreter_label] = auc
        plot_precision_recall_curve(precision, recall, y_true, ax, interpreter_label)
    
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')    
    model = torch.nn.Sequential(model, torch.nn.Linear(2048, 2))
    model.load_state_dict(torch.load(r"runs\dino\resnet50\384\LR0.01\model_params.pt"))
    model = model.to(device)

    interpreter = GradCAM(model=model, target_layer=model[0].layer4[-1], use_cuda=True)
    interpreter_label = "GradCAM\n(ResNet50-DINO)"
    mass_threshold = 0.8
    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        output_visualisation_column(dataset, interpreter, interpreter_label, axes[:,col_num], num_samples)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter)
    elif experiment == "AUC":
        y_true, y_pred = get_labels_and_pred_array(dataset, interpreter)
        precision, recall, thresholds, auc = calc_pre_rec_curve(y_true, y_pred)
        AUC[interpreter_label] = auc
        plot_precision_recall_curve(precision, recall, y_true, ax, interpreter_label)

    interpreter = get_random_map
    interpreter_label = "Random"
    if experiment == "intersect bar chart":
        hit_rates.append(get_hit_rate(dataset, interpreter))
        sensitivities.append(calc_weighted_sensitivity(dataset, interpreter))
        methods.append(interpreter_label)
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)
   
    if experiment == "intersect bar chart":
        df = pd.DataFrame(list(zip(methods, hit_rates, sensitivities)), columns =['Method', 'Top Pixel Hit Rate', "Weighted Sensitivity"])
        print(df)
        df.plot.bar(x="Method", rot=0)
        plt.xlabel("Method (Model)")
        plt.ylabel("Mean metric score")
        plt.show()
    elif experiment == "side by side vis":
        for idx in range(num_samples):
            image, seg, seg_cum, _ = dataset[idx]
            visualisation.imshow(image, ax=axes[idx, 0])
            visualisation.imshow(image, ax=axes[idx, 1])
            axes[idx, 0].imshow(cv2.resize(seg, dsize=(image.size(-1), image.size(-1)), interpolation=cv2.INTER_NEAREST), alpha=0.5)
            axes[idx, 1].imshow(cv2.resize(seg_cum, dsize=(image.size(-1), image.size(-1)), interpolation=cv2.INTER_NEAREST), alpha=0.5)
            axes[idx, 0].set_xticks([])
            axes[idx, 0].set_yticks([])
            axes[idx, 1].set_xticks([])
            axes[idx, 1].set_yticks([])
        axes[0, 0].title.set_text("Ground Truth")
        axes[0, 1].title.set_text("Ground Truth Patched")
        plt.show()
    elif experiment == "feature detected":
        # intersect_by_type = np.zeros(4)
        # for idx in range(len(datasets["test"])):
        #     _, _, seg_patch, _ = datasets["test"][idx]
        #     intersect_by_type[:-1] += np.count_nonzero(seg_patch, (0,1))
        # feature_proportions["Groundtruth"] = intersect_by_type/intersect_by_type.sum()

        X = np.arange(len(list(feature_proportions.values())[0]))
        width = 0.8 / len(feature_proportions)
        for idx, (method_name, proportions), in enumerate(feature_proportions.items()):
            plt.bar(X+idx*width, proportions, width=width, label=method_name)
        
        displacement = width*(len(X)+1)/2
        plt.xticks(X+displacement, ["Hard Exudates", "Haemorrhages", "Microaneurysms", "Other"])
        plt.ylabel("Fraction of identified features")
        plt.legend()
        # plt.show()
    elif experiment == "AUC":
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, bbox_to_anchor=(1.04,0.7), loc="upper left")
        plt.show()
    # for idx in range(len(datasets["test"])):
    #     print(idx)
    
    # visualise_sample_heatmap(model, interpreter, datasets["test"][4], 1)

    # fig, axes = plt.subplots(1, 3)
    # idx = 55
    # image, _, seg_cum, _ = dataset[idx]
    # heatmap, intercept = generate_heatmaps(dataset[idx], interpreter)
    # for ax in axes:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     visualisation. imshow(image, ax=ax)
    # axes[0].imshow(cv2.resize(seg_cum.max(-1), dsize=(image.size(-1), image.size(-1)), interpolation=cv2.INTER_NEAREST), alpha=0.5)
    # axes[1].imshow(cv2.resize(heatmap, dsize=(image.size(-1), image.size(-1)), interpolation=cv2.INTER_NEAREST), alpha=0.5)
    # axes[2].imshow(cv2.resize(intercept.max(-1), dsize=(image.size(-1), image.size(-1)), interpolation=cv2.INTER_NEAREST), alpha=0.5)#, vmin=0, vmax=0.02)
    # plt.show()