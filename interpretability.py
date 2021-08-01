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

def visualise_sample_heatmap(model, interpreter, sample, mass_threshold):
    image, seg, seg_cum, _ = sample
    heatmap, heatmap_threshold, intersect_map = generate_heatmaps(sample, interpreter, mass_threshold)

    fig, axes = plt.subplots(1,6)
    titles = ["Input", "GT Annotation", "GT Annotation Patched", "Heatmap", "Threshold Heatmap", "Intersect"]
    for ax, title in zip(axes, titles):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title)
        visualisation.imshow(image, ax=ax)

    img_dim = image.size(-1)
    axes[1].imshow(seg, alpha=0.5)
    axes[2].imshow(cv2.resize(seg_cum, dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5)
    axes[3].imshow(cv2.resize(heatmap, dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5)
    axes[4].imshow(cv2.resize(heatmap_threshold, dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5)
    axes[5].imshow(cv2.resize(intersect_map.max(-1), dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5)
    
    plt.show()

def calc_intersect_map(seg_cum, heatmap):
    intersect_map = np.multiply(seg_cum, heatmap[:,:,np.newaxis])
    return intersect_map

def get_patch_heatmap(heatmap, patch_size=16):
    num_patches = heatmap.shape[-1] // patch_size
    return heatmap.reshape(num_patches, patch_size, num_patches, patch_size).sum(axis=(1, 3))

def threshold_heatmap(heatmap, mass_threshold):
    if mass_threshold < 1:
        heatmap_sorted = np.sort(heatmap.ravel())
        total_mass = heatmap.sum()
        heatmap_cum_sum = heatmap_sorted.cumsum()
        greater_than_mass_threshold = heatmap_cum_sum > total_mass*(1-mass_threshold)
        threshold = heatmap_sorted[greater_than_mass_threshold.searchsorted(True)]
        heatmap = np.where(heatmap>=threshold, heatmap, 0)
    heatmap /= heatmap.sum()
    return heatmap

def get_avg_intersection_proportion(dataset, interpreter, threshold, device):
    intersect_proportion = np.zeros(len(dataset))
    for idx in range(len(dataset)):
        image, seg, seg_cum, _ = dataset[idx]
        heatmap, heatmap_threshold, intersect_map = generate_heatmaps(dataset[idx], interpreter, threshold, device)
        intersect_proportion[idx] = intersect_map.max(-1).sum() 
    intersect_mean = intersect_proportion.mean()
    intersect_std_err = intersect_proportion.std()/math.sqrt(len(dataset))
    return intersect_mean, intersect_std_err

def get_random_map(image_batch):
    num_patches = image_batch.size(-1)//16
    return np.random.rand(num_patches, num_patches)

def plot_av_intercept_against_thresholds(dataset, interpreter, thresholds, label):
    mean_log = np.zeros(len(thresholds))
    err_log = np.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        mean_log[idx], err_log[idx] = get_avg_intersection_proportion(datasets["train"], interpreter, threshold, device)
    plt.errorbar(thresholds, mean_log, yerr=err_log, label=label)

def map_attention_of_class_token_last_layer(last_layer_attentions):
    class_attentions = last_layer_attentions[0, :, 0, 1:]
    width = int(class_attentions.size(-1)**0.5)
    class_attentions = class_attentions.reshape(-1, width, width).numpy()
    class_attentions = class_attentions.sum(0)
    return class_attentions


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

def generate_heatmaps(sample, interpreter, mass_threshold):
    image, _, seg_cum, _ = sample
    heatmap = interpreter(image.unsqueeze(0).to(device))
    if not (heatmap.shape == seg_cum.shape[:-1]):
            heatmap = get_patch_heatmap(heatmap[0])
    heatmap_threshold = threshold_heatmap(heatmap, mass_threshold)
    intersect_map = calc_intersect_map(seg_cum, heatmap_threshold)
    return heatmap, heatmap_threshold, intersect_map

def output_visualisation_column(image, heatmap, heatmap_threshold, intersect_map, col, axes, title):
    axes[0, col].title.set_text(title)
    for ax in axes[:,col]:
        ax.set_xticks([])
        ax.set_yticks([])
        visualisation.imshow(image, ax=ax)

    img_dim = image.size(-1)
    axes[0, col].imshow(cv2.resize(heatmap/heatmap.sum(), dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5, vmin=0, vmax=0.02)
    axes[1, col].imshow(cv2.resize(heatmap_threshold, dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5, vmin=0, vmax=0.1)
    axes[2, col].imshow(cv2.resize(intersect_map.max(-1), dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5, vmin=0, vmax=0.1)
    
def calc_intersection_by_type(dataset, interpreter, mass_threshold):
    intersect_by_type = np.zeros(4)
    for idx in range(len(dataset)):
        _, heatmap_threshold, intersect_map = generate_heatmaps(dataset[idx], interpreter, mass_threshold)
        intersect_by_type[:-1] += np.count_nonzero(intersect_map, (0,1))
        intersect_by_type[-1] += np.count_nonzero(heatmap_threshold) - np.count_nonzero(intersect_map.max(-1))
    return intersect_by_type/intersect_by_type.sum()

if __name__ == "__main__":
    model_dir_ViT = r"runs\384_Strong_Aug\vit_small_patch16_224_in21k\LR_0.01"
    model_dir_BiT = r"runs\384_Strong_Aug\resnetv2_50x1_bitm_in21k\LR_0.01"

    np.random.seed(13)
    full_dataset = IDRiD_Dataset(r'data\idrid', img_size=384)
    class_names = ["Healthy", "Refer"]
    dataset_names = ["train", "test"]    
    dataset_proportions = np.array([0.5, 0.5])
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    experiment = "feature detected"

    if experiment == "threshold search":
        thresholds = 0.1 * np.arange(1, 2)
    elif experiment == "intersect bar chart":
        intersects = []
        intersect_error = []
        methods = []
    elif experiment == "side by side vis":
        fig, axes = plt.subplots(3, 6, squeeze=False)
        col_num = 1
        vis_sample = datasets["test"][4]
    elif experiment == "feature detected":
        feature_proportions = {}

    
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')    
    model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
    model.load_state_dict(torch.load(r"runs\dino_EyePACS_Dataset_07_28_12_18_42\model_params.pt"))
    model = model.to(device)

    

    interpreter = Last_Layer(model[0])
    interpreter_label = "Last Layer (DINO)"
    mass_threshold = 0.1

    if experiment == "threshold search":
        plot_av_intercept_against_thresholds(datasets["train"], interpreter, thresholds, interpreter_label)
    elif experiment == "intersect bar chart":
        intersect_mean, intersect_std_err = get_avg_intersection_proportion(datasets["test"], interpreter, mass_threshold, device)
        intersects.append(intersect_mean)
        intersect_error.append(intersect_std_err)
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        heatmap, heatmap_threshold, intersect_map = generate_heatmaps(vis_sample, interpreter, mass_threshold)
        output_visualisation_column(vis_sample[0], heatmap, heatmap_threshold, intersect_map, col_num, axes, interpreter_label)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')    
    model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
    model.load_state_dict(torch.load(r"runs\dino_EyePACS_Dataset_07_28_12_18_42\model_params.pt"))
    model = model.to(device)

    interpreter = VITAttentionRollout(model[0], discard_ratio=0.9, head_fusion="max")
    interpreter_label = "Attention Rollout (DINO)"
    mass_threshold = 0.1
    if experiment == "threshold search":
        plot_av_intercept_against_thresholds(datasets["train"], interpreter, thresholds, interpreter_label)
    elif experiment == "intersect bar chart":
        intersect_mean, intersect_std_err = get_avg_intersection_proportion(datasets["test"], interpreter, mass_threshold, device)
        intersects.append(intersect_mean)
        intersect_error.append(intersect_std_err)
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        heatmap, heatmap_threshold, intersect_map = generate_heatmaps(vis_sample, interpreter, mass_threshold)
        output_visualisation_column(vis_sample[0], heatmap, heatmap_threshold, intersect_map, col_num, axes, interpreter_label)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)

    model = evaluate.load_model("vit_small_patch16_224_in21k", device, class_names, model_dir_ViT, 384)  
    interpreter = Last_Layer(model)
    interpreter_label = "Last Layer (ViT)"
    mass_threshold = 0.5
    if experiment == "threshold search":
        plot_av_intercept_against_thresholds(datasets["train"], interpreter, thresholds, interpreter_label)
    elif experiment == "intersect bar chart":
        intersect_mean, intersect_std_err = get_avg_intersection_proportion(datasets["test"], interpreter, mass_threshold, device)
        intersects.append(intersect_mean)
        intersect_error.append(intersect_std_err)
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        heatmap, heatmap_threshold, intersect_map = generate_heatmaps(vis_sample, interpreter, mass_threshold)
        output_visualisation_column(vis_sample[0], heatmap, heatmap_threshold, intersect_map, col_num, axes, interpreter_label)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)
    
    model = evaluate.load_model("vit_small_patch16_224_in21k", device, class_names, model_dir_ViT, 384)  
    interpreter = VITAttentionRollout(model, discard_ratio=0.9, head_fusion="max")
    interpreter_label = "Attention Rollout (ViT)"
    mass_threshold = 0.3
    if experiment == "threshold search":
        plot_av_intercept_against_thresholds(datasets["train"], interpreter, thresholds, interpreter_label)
    elif experiment == "intersect bar chart":
        intersect_mean, intersect_std_err = get_avg_intersection_proportion(datasets["test"], interpreter, mass_threshold, device)
        intersects.append(intersect_mean)
        intersect_error.append(intersect_std_err)
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        heatmap, heatmap_threshold, intersect_map = generate_heatmaps(vis_sample, interpreter, mass_threshold)
        output_visualisation_column(vis_sample[0], heatmap, heatmap_threshold, intersect_map, col_num, axes, interpreter_label)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)
    
    model = evaluate.load_model("resnetv2_50x1_bitm_in21k", device, class_names, model_dir_BiT)   
    interpreter = GradCAM(model=model, target_layer=model.stages[-1], use_cuda=True)
    interpreter_label = "GradCAM (BiT)"
    mass_threshold = 0.1
    if experiment == "threshold search":
        plot_av_intercept_against_thresholds(datasets["train"], interpreter, thresholds, interpreter_label)
    elif experiment == "intersect bar chart":
        intersect_mean, intersect_std_err = get_avg_intersection_proportion(datasets["test"], interpreter, mass_threshold, device)
        intersects.append(intersect_mean)
        intersect_error.append(intersect_std_err)
        methods.append(interpreter_label)
    elif experiment == "side by side vis":
        heatmap, heatmap_threshold, intersect_map = generate_heatmaps(vis_sample, interpreter, mass_threshold)
        output_visualisation_column(vis_sample[0], heatmap, heatmap_threshold, intersect_map, col_num, axes, interpreter_label)
        col_num += 1
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)
    
    interpreter = get_random_map
    interpreter_label = "Random"
    mass_threshold = 1
    if experiment == "threshold search":
        plot_av_intercept_against_thresholds(datasets["train"], interpreter, thresholds, interpreter_label)
    elif experiment == "intersect bar chart":
        intersect_mean, intersect_std_err = get_avg_intersection_proportion(datasets["test"], interpreter, mass_threshold, device)
        intersects.append(intersect_mean)
        intersect_error.append(intersect_std_err)
        methods.append(interpreter_label)
    elif experiment == "feature detected":
        feature_proportions[interpreter_label] = calc_intersection_by_type(datasets["test"], interpreter, mass_threshold)
   
    print(feature_proportions)

    if experiment == "threshold search":
        plt.xlabel("Mass threshold")
        plt.ylabel("Intercept proportion")
        plt.legend()
        plt.show()
    elif experiment == "intersect bar chart":
        y_pos = np.arange(len(methods))
        plt.bar(y_pos, intersects, yerr=intersect_error)
        plt.xticks(y_pos, methods)
        plt.ylabel("Intersect proportion")
        plt.show()
    elif experiment == "side by side vis":
        titles_gt = ["Input", "Semgentation Map", "Patched Semgentation Map"]
        image, seg, seg_cum, _ = vis_sample
        for ax, title in zip(axes[:, 0], titles_gt):
            ax.set_ylabel(title)
            visualisation.imshow(image, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
        img_dim = image.size(-1)
        axes[1, 0].imshow(seg, alpha=0.5)
        axes[2, 0].imshow(cv2.resize(seg_cum, dsize=(img_dim, img_dim), interpolation=cv2.INTER_NEAREST), alpha=0.5)
        axes[0, 0].title.set_text("Ground truth")

        titles_heatmap = ["Heatmap", "Threshold Heatmap", "Intersect"]
        for ax, title in zip(axes[:, 1], titles_heatmap):
            ax.set_ylabel(title)
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
        plt.show()
    # for idx in range(len(datasets["test"])):
    #     print(idx)
    
    # visualise_sample_heatmap(model, interpreter, datasets["test"][4], 1)