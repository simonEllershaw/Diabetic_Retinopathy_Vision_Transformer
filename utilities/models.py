import copy
import math
import timm
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_name, pretraining, num_classes, img_size=224, model_fpath=None):
    if pretraining == "DINO":
        if model_name == "ResNet50":
            model = torch.hub.load('facebookresearch/dino:main', "dino_resnet50")
            model = torch.nn.Sequential(model, torch.nn.Linear(2048, 2))
        elif model_name == "ViT-S":
            model = torch.hub.load('facebookresearch/dino:main', "dino_vits16")        
            model = torch.nn.Sequential(model, torch.nn.Linear(model.num_features, 2))
        use_inception_norm = False
    elif pretraining == "21k":
        if model_name == "ResNet50":
            model = timm.create_model("resnetv2_50x1_bitm_in21k", pretrained=True, num_classes=num_classes)
        elif model_name == "ViT-S":
            model = timm.create_model("vit_small_patch16_224_in21k", pretrained=True, num_classes=num_classes)
            if img_size != 224:
                model = resize_ViT(model, img_size)
        use_inception_norm = True
    if model_fpath is not None:
        model.load_state_dict(torch.load(model_fpath))
    return model, use_inception_norm

def resize_ViT(model, new_input_size):
    new_patch_embed = timm.models.layers.patch_embed.PatchEmbed(img_size=384)
    new_patch_embed.proj = copy.deepcopy(model.patch_embed.proj)
    model.patch_embed = new_patch_embed

    num_patches = model.patch_embed.num_patches
    pos_embed_new = torch.nn.Parameter(torch.zeros(1, num_patches + model.num_tokens, model.embed_dim))
    pos_embed_new = timm.models.vision_transformer.resize_pos_embed(model.pos_embed, pos_embed_new)
    model.pos_embed = torch.nn.Parameter(pos_embed_new)
    return model

def calc_pos_embed_similarites(pos_embed, stride=0):
    patch_pos_embed = pos_embed[0, 1:].detach()
    if stride > 0:
        # Arrange into 2D matrix
        num_patch_embeddings = patch_pos_embed.size(0)
        len_square = int(math.sqrt(num_patch_embeddings))
        patch_pos_embed = patch_pos_embed.reshape((len_square,len_square,-1))
        # Stride over columns and rows
        patch_pos_embed = patch_pos_embed[::stride,::stride]
        # Flatten
        patch_pos_embed = patch_pos_embed.reshape(((len_square//2)**2,-1))
    # Init similarity variables
    num_patch_embeddings = patch_pos_embed.size(0)
    cos_sim = torch.zeros((num_patch_embeddings, num_patch_embeddings))
    cos = torch.nn.CosineSimilarity(dim=0)

    # Calc simliarty of each pos_embed with every other pos_embed
    for idx_main, patch_embedding_main in enumerate(patch_pos_embed):
        for idx_compare, patch_embedding_compare in enumerate(patch_pos_embed):
            cos_sim[idx_main, idx_compare] = cos(patch_embedding_main, patch_embedding_compare).detach()
    
    # Reshape into 2D matrix
    length_patch_square = int(math.sqrt(num_patch_embeddings))
    cos_sim = cos_sim.reshape((length_patch_square, length_patch_square, length_patch_square, length_patch_square))
    return cos_sim

def visualise_postional_embeddings(cos_sim):
    # Init plot
    length_patch_square = cos_sim.size(0)
    fig, axs = plt.subplots(length_patch_square, length_patch_square)
    cbar_ax = fig.add_axes([.88, .15, .02, .7])
    fig.subplots_adjust(right=0.85)

    # For each pos embed plot heatmap of it's similarites
    for row in range(length_patch_square):
        for column in range(length_patch_square):
            ax = axs[row, column]
            sns.heatmap(cos_sim[row,column], 
                        ax=ax, 
                        vmin=-1, 
                        vmax=1, 
                        cbar=row+column== 0, 
                        xticklabels=False, 
                        yticklabels=False, 
                        square=True, 
                        cbar_ax=cbar_ax,
                        cbar_kws={'label': 'Cosine Similarity'})
            if column == 0:
                plt.setp(ax, ylabel=row+1)
            if row == length_patch_square-1:
                plt.setp(ax, xlabel=column+1)   
    plt.show()


if __name__ == "__main__":
    model = torch.hub.load('facebookresearch/dino:main', "dino_resnet50")
    exit()
    inp = torch.ones((3, 3, 384, 384))*0.5
    model = timm.create_model("vit_small_patch16_224_in21k", pretrained=True, num_classes=2)
    # model = resize_ViT(model, 384)
    # model.load_state_dict(torch.load(r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01\model_params.pt"))
    
    # cos_sim = calc_pos_embed_similarites(model.pos_embed, stride=2)
    # visualise_postional_embeddings(cos_sim)

    # print(model.pos_embed.size())

    weights = model.patch_embed.proj.weight
    print(weights.size())
    idx = 0
    test = weights[idx].detach().numpy().transpose(1, 2, 0)/torch.max(weights[idx]).item()
    print(torch.max(test))
    print(test.shape)
    plt.imshow(test)
    plt.show()