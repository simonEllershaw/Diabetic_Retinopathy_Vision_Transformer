import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision
import torch
import inference
import itertools

def imshow(inp, ax, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)

def sample_batch(dataloader, class_names, model=None, device=None, num_samples=4):
    fig, axs = plt.subplots(1, num_samples, sharey=True)
    images, labels, fnames = next(iter(dataloader))
    if model is not None:
        preds, probs = inference.images_to_probs(model, images.to(device))
    for idx in range(num_samples):
        imshow(images[idx], axs[idx])
        if model is None:
            title= class_names[labels[idx]] + " " + fnames[idx]
            color="black"
        else:
            title = f"{class_names[preds[idx]]}({probs[idx] * 100.0:.1f}%) \n label:{class_names[labels[idx]]}"
            color=("green" if preds[idx]==labels[idx].item() else "red")
        axs[idx].set_title(title, color=color)
    return fig

def plot_confusion_matrix(cm, class_names):
    # https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    cm_fract = cm/(cm.sum(1, keepdim=True))
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm_fract, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
       
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f"{cm[i, j].item()}({cm_fract[i, j].item():.2f})", horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

if __name__ == "__main__":
    cm = torch.tensor([[1,4],[3,3]])
    plot_confusion_matrix(cm, ["0", "1"])
    plt.show()