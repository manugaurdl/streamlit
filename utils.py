from PIL import Image
import streamlit as st
import pandas as pd
import os
import torch
import sys
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

@st.cache_data
def load_ade_data(data_dir):
    idx2class = {int(_.split("\t")[0]) : _.split("\t")[-1][:-1].strip() for _ in open(os.path.join(data_dir, "objectInfo150.txt"), "r").readlines()[1:]} #0=background, 1=wall
    ann_dir = os.path.join(data_dir, "annotations/validation")
    files = os.listdir(ann_dir)
    class2file = defaultdict(list)
    for f in files:
        file = os.path.join(ann_dir, f)
        seg = np.array(Image.open(file))
        seg = seg[seg!=0]
        for obj_class in np.unique(seg):
            class2file[idx2class[obj_class]].append(f)

    return class2file

@st.cache_data
def load_df(data_path):
    csv_path = os.path.join(data_path)
    return pd.read_csv(csv_path)

@st.cache_resource
def init_model(img_size, patch_size, ckpt_dir, ckpt, repo_path):
    sys.path.append(repo_path)
    from src.models import FlamingoCrossAttn
    from src.training import lora_siglip
    from src.dataloaders.utils import eval_transform

    msg = st.empty()
    msg.text("| LOADING MODEL...")
    encoder = FlamingoCrossAttn(
                visual_encoder="siglip_vitl",
                text_encoder ="roberta",
                img_res = img_size,
                patch_size = patch_size,
                cross_attn_layers =[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
                cross_attn_ffn_mult =2,
                )

    lora_siglip(
        encoder,
        rank = 8,
        last_n_blocks=6
        )
    trained_wts = torch.load(os.path.join(ckpt_dir, ckpt))['model_state_dict']
    trained_wts = {k.replace('visual_encoder.siglip.visual.trunk', 'visual_encoder.trunk'): v for k,v in trained_wts.items() if 'visual_encoder.siglip.visual.trunk'}

    state_dict = {}
    for k,v in encoder.state_dict().items():
        if k in trained_wts: 
            state_dict[k] = trained_wts[k]
        else:
            state_dict[k] = v

    encoder.load_state_dict(state_dict)
    msg.empty()
    # print(f"no of trained weight : {len(trained_wts)}")
    st.success("Model loaded!")
    
    return encoder, eval_transform(img_size)


@st.cache_resource
def roberta_encode(texts, device):
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval();roberta.to(device)
    class2feat= dict(zip(texts, [roberta.extract_features(roberta.encode(text.lower().strip())) for text in texts]))
    roberta = None
    torch.cuda.empty_cache()
    return class2feat


def plot(image):
    img = Image.open(image_path)
    st.image(img)


def viz_heatmap(img: Image.Image, gt_map: torch.Tensor, pred_map: torch.Tensor):
    """
    img: PIL Image
    gt_map, pred_map: torch.Tensor of shape (16,16)
    """

    _spacer, c1, c2, c3,  = st.columns([0.3, 1, 1, 1])
    c1.markdown("##### Image")
    c2.markdown("##### GT heatmap")
    c3.markdown("##### Predicted")
    H = W = 224
    # 1) Resize image
    img_resized = img.resize((H, W))
    img_arr = np.array(img_resized)
    
    # 2) Prepare heatmaps
    if gt_map is not None:
        gt_tensor = gt_map.float().unsqueeze(0).unsqueeze(0)      # 1×1×16×16
        
        gt_resized = (
            F.interpolate(gt_tensor, size=(H, W), mode='bilinear', align_corners=False)
            .squeeze()
            .numpy()
        )
    pred_tensor = pred_map.float().unsqueeze(0).unsqueeze(0)  # 1×1×16×16
    pred_resized = (
        F.interpolate(pred_tensor, size=(H, W), mode='bilinear', align_corners=False)
        .squeeze()
        .numpy()
    )
    
    # 3) Plot
    num_plots = 2 if gt_map is None else 3
    fig, axes = plt.subplots(1, num_plots)
    
    # Original
    axes[0].imshow(img_arr)
    # axes[0].set_title("Original")
    axes[0].axis("off")
    
    # GT overlay
    if gt_map is not None:
        axes[1].imshow(img_arr)
        hm1 = axes[1].imshow(gt_resized, cmap="jet", alpha=0.5)
        # fig.colorbar(hm1, ax=axes[1], fraction=0.046, pad=0.04)
        # axes[1].set_title("Ground Truth")
        axes[1].axis("off")
    
    plot_idx = 1 if gt_map is None else 2
    # Pred overlay
    axes[plot_idx].imshow(img_arr)
    hm2 = axes[plot_idx].imshow(pred_resized, cmap="jet", alpha=0.5)
    # fig.colorbar(hm2, ax=axes[2], fraction=0.046, pad=0.04)
    # axes[2].set_title("Predicted")
    axes[plot_idx].axis("off")
    
    fig.tight_layout()
    return fig


# def plot_attn_maps(attn_map):

#     attn_map = attn_map.mean(dim=1) # sum over all queries
#     nh = attn_map.shape[0]
#     threshold = 0.6
#     w_featmap = 13
#     h_featmap = 13
    
#     # we keep only a certain percentage of the mass
#     val, idx = torch.sort(attn_map) # q.K in ascending order
#     val /= torch.sum(val, dim=1, keepdim=True) # conver to pdf; since we omit cls key, its not a pdf anymore
#     cumval = torch.cumsum(val, dim=1) # for each head, cumval[i,j] tells how much p_mass is in the bottom half "j" patches 
#     th_attn = cumval > (1 - threshold) # get patches that have attn_mass within the top end of the distribution
#     idx2 = torch.argsort(idx)
#     for head in range(nh):
#         th_attn[head] = th_attn[head][idx2[head]]
#     th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
#     # interpolate
#     th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=14, mode="nearest")[0].cpu().numpy()
#     attn_map = attn_map.reshape(nh, w_featmap, h_featmap)
#     attn_map = F.interpolate(attn_map.unsqueeze(0), scale_factor=14, mode="nearest")[0].cpu()
#     attn_map = attn_map.detach().numpy()
    
#     # show and save attn_map heatmaps
#     # torchvision.utils.save_image(torchvision.utils.make_grid(pixel_values, normalize=True, scale_each=True), os.path.join(output_dir, "img.png"))
    
#     fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # Adjust figsize as needed
    
#     for j in range(nh):
#         row, col = divmod(j, 4)
#         axs[row, col].imshow(attn_map[j], cmap='viridis')  # You can change colormap if needed
#         axs[row, col].set_title(f"Head {j}")
#         axs[row, col].axis('off')  # Hide axes for a cleaner look
    
#     plt.tight_layout()
#     return fig