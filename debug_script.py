import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import random
import torch
from PIL import Image
import pandas as pd

# from utils import load_df, init_model, viz_heatmap#, plot_attn_maps


def load_df(data_dir):
    csv_path = os.path.join(data_dir, "referit/refCOCOg", "test.csv")
    return pd.read_csv(csv_path)


def init_model(img_size, patch_size, ckpt_dir, ckpt):
    import sys
    sys.path.append("/home/manugaur/mllm_inversion")
    from src.models import FlamingoCrossAttn
    from src.training import lora_siglip
    from src.dataloaders.utils import eval_transform

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
    return encoder, eval_transform(img_size)



#load  refCOCO
data_dir = "/storage/users/manugaur/mllm_inversion"
if not os.path.isdir(data_dir): 
    data_dir = "/workspace/manugaur/mllm_inversion"
df = load_df(data_dir)

#args
ckpt_dir = os.path.join(data_dir, "checkpoints")
ckpt = ""
device = "cuda"
img_res = 182
patch_size = 14
N = img_res//14
text_feats_dir = os.path.join(data_dir, "referit/refCOCOg/roberta_feats")
#boolean args
TEXT_INPUT = True

# refCOCO GT
softCE_targets = torch.load(os.path.join(os.path.join(data_dir, f"referit/refCOCOg/gauss_maps_256/{N}x{N}_image", f"test.pt"))) #cache them

model, transform = init_model(img_res, patch_size, ckpt_dir, ckpt)
model.to(device)
model.eval()


### idx to images that weren't seen during training
proxy_idx = [6, 7, 104, 105, 190, 191, 220, 221, 236, 237, 291, 292, 305, 365, 366, 391, 392, 405, 406, 432, 433, 497, 498, 708, 709, 729, 730, 750, 751, 779, 780, 797, 798, 846, 847, 858, 859, 860, 942, 943, 975, 976, 995, 996, 999, 1000, 1027, 1028, 1142, 1143, 1280, 1281, 1336, 1337, 1375, 1376, 1402, 1403, 1460, 1461, 1537, 1538, 1600, 1601, 1692, 1693, 1710, 1711, 1831, 1832, 1861, 1862, 1904, 1905, 1910, 1911, 1914, 1915, 2157, 2158, 2182, 2183, 2233, 2234, 2293, 2294, 2359, 2360, 2372, 2373, 2388, 2389, 2435, 2436, 2447, 2448, 2584, 2585, 2600, 2601, 2610, 2611, 2614, 2738, 2739, 2872, 2873, 2975, 2976, 3006, 3007, 3166, 3167, 3369, 3370, 3452, 3453, 3473, 3474, 3508, 3509, 3565, 3566, 3608, 3609, 3650, 3651, 3749, 3750, 3775, 3776, 3794, 3795, 3844, 3845, 3846, 3847, 3902, 3903, 3987, 3988, 3991, 3992, 4081, 4248, 4249, 4262, 4263, 4307, 4308, 4311, 4312, 4450, 4451, 4454, 4455, 4471, 4472, 4477, 4478, 4517, 4518, 4554, 4555, 4593, 4594, 4699, 4700, 4719, 4720, 4787, 4788, 4796, 4797, 4831, 4832, 4833, 4834, 4847, 4848, 4886, 4887, 4978, 4979, 5034, 5035, 5065, 5170, 5171, 5252, 5253, 5318, 5319, 5363, 5364, 5493, 5494, 5511, 5512, 5527, 5528, 5576, 5577, 5619, 5620, 5621, 5622, 5623, 5624, 5634, 5635, 5663, 5664, 5723, 5724, 5776, 5777, 5883, 5884, 5897, 5898, 5995, 5996, 6036, 6037, 6054, 6200, 6201, 6204, 6205, 6323, 6324, 6327, 6328, 6389, 6390, 6423, 6424, 6453, 6454, 6517, 6518, 6589, 6590, 6635, 6636, 6825, 6826, 6840, 6841, 6952, 6953, 6958, 6959, 7014, 7015, 7069, 7070, 7179, 7180, 7181, 7182, 7233, 7234, 7313, 7314, 7380, 7381, 7386, 7387, 7392, 7407, 7408, 7423, 7424, 7425, 7426, 7434, 7435, 7471, 7472, 7526, 7527, 7602, 7603, 7607, 7608, 7626, 7669, 7717, 7762, 7814, 7815, 7975, 7976, 8000, 8001, 8002, 8003, 8064, 8065, 8073, 8074, 8118, 8119, 8150, 8151, 8199, 8200, 8261, 8262, 8301, 8302, 8430, 8457, 8458, 8507, 8508, 8575, 8576, 8592, 8611, 8612, 8680, 8681, 8702, 8703, 8768, 8769, 8775, 8776, 8911, 8912, 8917, 8918, 8972, 8973, 8976, 8977, 9023, 9024, 9080, 9120, 9139, 9140, 9167, 9168, 9191, 9192, 9266, 9267, 9272, 9273, 9331, 9332, 9411, 9412, 9415, 9416, 9549, 9550, 9584, 9585]


# if 'idx' not in st.session_state:
#     st.session_state.idx = 0
#     st.session_state.proxy_idx = proxy_idx[st.session_state.idx]
#     st.session_state.image_path = df.iloc[st.session_state.proxy_idx]["img_path"]
#     st.session_state.caps = df.iloc[st.session_state.proxy_idx]["captions"]

# col1, col2 = st.columns([1, 8])

# with col1:
#     if st.button("Prev", key="prev_btn"):
#         st.session_state.idx -= 1
#         st.session_state.proxy_idx = proxy_idx[st.session_state.idx]
#         st.session_state.image_path = df.iloc[st.session_state.proxy_idx]["img_path"]
#         st.session_state.caps = df.iloc[st.session_state.proxy_idx]["captions"]

# with col2:
#     if st.button("Next", key="next_btn"):
#         st.session_state.idx += 1
#         st.session_state.proxy_idx = proxy_idx[st.session_state.idx]
#         st.session_state.image_path = df.iloc[st.session_state.proxy_idx]["img_path"]
#         st.session_state.caps = df.iloc[st.session_state.proxy_idx]["captions"]


####args 
proxy_idx=0
image_path = df.iloc[proxy_idx]["img_path"]


#load precomputed text_feats
text_feat = torch.load(os.path.join(text_feats_dir, 'test', f"{proxy_idx}.pt")).to(device).unsqueeze(0)
attn_mask = torch.ones((1,text_feat.size(1))).bool().to(text_feat.device)

gt = softCE_targets[proxy_idx]
gt_heatmap = gt.view((N,N))
image = Image.open(image_path).convert('RGB')

#get model output
out_softce, out_bce, cos_sim_pos_negs = model(transform(image).unsqueeze(0).to(device), text_feat, attn_mask, n_pos=1)
output_heatmap = out_softce.softmax(dim=-1).view(N,N)
# plot outputs
fig = viz_heatmap(image, gt_heatmap, output_heatmap.detach().cpu())

# attn_map = model.visual_encoder.trunk.blocks[-1].attn.attn_map[0]
# attn_fig = plot_attn_maps(attn_map)
# st.pyplot(attn_fig, use_container_width=True)
