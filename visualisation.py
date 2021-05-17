import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision
import torch
import inference

def imshow(inp, ax, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)

def sample_batch(dataloader, class_names, model=None, device=None, num_samples=4):
    fig, axs = plt.subplots(1, num_samples, sharey=True)
    images, labels = next(iter(dataloader))
    if model is not None:
        preds, probs = inference.images_to_probs(model, images.to(device))
    for idx in range(num_samples):
        imshow(images[idx], axs[idx])
        if model is None:
            title= class_names[labels[idx]]
            color="black"
        else:
            title = f"{class_names[preds[idx]]}({probs[idx] * 100.0:.1f}%) \n label:{class_names[labels[idx]]}"
            color=("green" if preds[idx]==labels[idx].item() else "red")
        axs[idx].set_title(title, color=color)
    return fig