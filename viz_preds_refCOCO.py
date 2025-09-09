import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import random
import streamlit as st
st.set_page_config(layout="wide")
import torch
from PIL import Image
from utils import plot, load_df, init_model, viz_heatmap#, plot_attn_maps
import json

st.title("Qualitative Analysis")
#load  refCOCO
data_dir = "/storage/users/manugaur/mllm_inversion"
if not os.path.isdir(data_dir): 
    data_dir = "/workspace/manugaur/mllm_inversion"
df = load_df(data_dir)

#args
ckpt_dir = os.path.join(data_dir, "checkpoints")
ckpt = "vitl14_2.4m_samples_182res_bsz42.pth"
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

#load roberta
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval();roberta.to(device)


### idx to images that weren't seen during training
proxy_idx = json.load(open("data/uncontaminated_image_ids.json", "r"))


if 'idx' not in st.session_state:
    st.session_state.idx = 0
    st.session_state.proxy_idx = proxy_idx[st.session_state.idx]
    st.session_state.image_path = df.iloc[st.session_state.proxy_idx]["img_path"]
    st.session_state.caps = df.iloc[st.session_state.proxy_idx]["captions"]
    # st.session_state.user_text = None


def go_to_next_image():
    st.session_state.idx +=1
    st.session_state.user_text = "" # shouldn't it be None

def go_to_prev_image():
    st.session_state.idx -=1
    st.session_state.user_text = ""

current_proxy_idx = proxy_idx[st.session_state.idx]
image_path = df.iloc[current_proxy_idx]["img_path"]
caps = df.iloc[current_proxy_idx]["captions"]

col1, col2 = st.columns([1, 8])

with col1:
    st.button("Prev", on_click=go_to_prev_image)

with col2:
    st.button("Next", on_click=go_to_next_image)

st.write("**GT Caption:**", st.session_state.caps)


#image inputs
gt = softCE_targets[st.session_state.proxy_idx]
gt_heatmap = gt.view((N,N))
image = Image.open(st.session_state.image_path).convert('RGB')
#load precomputed text_feats
precomp_text_feat = torch.load(os.path.join(text_feats_dir, 'test', f"{st.session_state.proxy_idx}.pt")).to(device).unsqueeze(0)

def text_to_heatmap(text_feat):        
        attn_mask = torch.ones((1,text_feat.size(1))).bool().to(text_feat.device)
        #get model output
        outputs = model(transform(image).unsqueeze(0).to(device), text_feat, attn_mask).squeeze(0)
        output_heatmap = outputs.softmax(dim=-1).view(N,N)
        # plot outputs
        fig = viz_heatmap(image, gt_heatmap, output_heatmap.detach().cpu())    
        st.pyplot(fig, use_container_width=True)


user_text = st.text_input(
    "Enter text", key="user_text"
)
st.markdown("---")

if st.session_state.user_text:
    default_text_feat = roberta.extract_features(roberta.encode(user_text.lower().strip()))
    text_to_heatmap(default_text_feat)
else:
    text_to_heatmap(precomp_text_feat)

# attn_map = model.visual_encoder.trunk.blocks[-1].attn.attn_map[0]
# attn_fig = plot_attn_maps(attn_map)
# st.pyplot(attn_fig, use_container_width=True)
