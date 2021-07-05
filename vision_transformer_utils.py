import timm
import torch
import copy


def resize_ViT(model, new_input_size):
    new_patch_embed = timm.models.layers.patch_embed.PatchEmbed(img_size=384)
    new_patch_embed.proj =  copy.deepcopy(model.patch_embed.proj)
    model.patch_embed = new_patch_embed

    num_patches = model.patch_embed.num_patches
    pos_embed_new = torch.nn.Parameter(torch.zeros(1, num_patches + model.num_tokens, model.embed_dim))
    pos_embed_new = timm.models.vision_transformer.resize_pos_embed(model.pos_embed, pos_embed_new)
    model.pos_embed = torch.nn.Parameter(pos_embed_new)
    return model

if __name__ = "__main__":
    inp = torch.ones((3, 3, 384, 384))*0.5
    model = timm.create_model("vit_small_patch16_224_in21k", pretrained=True, num_classes=5)
    model = resize_ViT(model, 384)
    print(model(inp).size())
