import torch
import numpy as np

def generate_heatmaps(dataset, interpreter, img_size, patch_size):
    heatmap_size = img_size//patch_size
    heatmaps = np.zeros((len(dataset), heatmap_size, heatmap_size))
    for idx in range(len(dataset)):
        heatmaps[idx,:,:] = generate_heatmap(dataset[idx], interpreter, patch_size)
    return heatmaps

def generate_heatmap(sample, interpreter, patch_size):
    image, _, seg_cum, _, _ = sample
    heatmap = interpreter(image.unsqueeze(0))
    if not (heatmap.shape == seg_cum.shape):
            heatmap = get_patch_heatmap(heatmap[0], patch_size)
    heatmap = heatmap/heatmap.sum()
    return heatmap

def get_patch_heatmap(heatmap, patch_size):
    num_patches = heatmap.shape[-1] // patch_size
    return heatmap.reshape(num_patches, patch_size, num_patches, patch_size).sum(axis=(1, 3))


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
            _ = self.model(input_tensor)
        return self.map_attention_of_class_token_last_layer(self.attentions[-1])

    def map_attention_of_class_token_last_layer(self, last_layer_attentions):
        class_attentions = last_layer_attentions[0, :, 0, 1:]
        width = int(class_attentions.size(-1)**0.5)
        class_attentions = class_attentions.reshape(-1, width, width).numpy()
        class_attentions = class_attentions.sum(0)
        return class_attentions


def get_random_map(image_batch):
    num_patches = image_batch.size(-1)//16
    return np.random.rand(num_patches, num_patches)