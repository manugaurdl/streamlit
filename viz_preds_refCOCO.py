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
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""

def clear_user_text():
    st.session_state.user_text = ""

cols = st.columns(1)
with cols[0]:
    st.session_state.idx = st.number_input(
        label="image index:",
        min_value=0,
        max_value=len(proxy_idx)-1,
        # value=st.session_state.idx, # Defaults to the current index
        step=1,
        on_change=clear_user_text,
    )
    user_text = ""


st.text_input(
    "Enter text", key="user_text"
)

st.session_state.current_proxy_idx = proxy_idx[st.session_state.idx]
st.session_state.image_path = df.iloc[st.session_state.current_proxy_idx]["img_path"]
st.session_state.caps = df.iloc[st.session_state.current_proxy_idx]["captions"]

st.write("**GT Caption:**", st.session_state.caps)
st.markdown("---")

#image inputs
gt = softCE_targets[st.session_state.current_proxy_idx]
gt_heatmap = gt.view((N,N))
image = Image.open(st.session_state.image_path).convert('RGB')
#load precomputed text_feats
precomp_text_feat = torch.load(os.path.join(text_feats_dir, 'test', f"{st.session_state.current_proxy_idx}.pt")).to(device).unsqueeze(0)

def text_to_heatmap(text_feat):        
        attn_mask = torch.ones((1,text_feat.size(1))).bool().to(text_feat.device)
        #get model output
        outputs = model(transform(image).unsqueeze(0).to(device), text_feat, attn_mask).squeeze(0)
        output_heatmap = outputs.softmax(dim=-1).view(N,N)
        # plot outputs
        fig = viz_heatmap(image, gt_heatmap, output_heatmap.detach().cpu())    
        st.pyplot(fig, use_container_width=True)



if st.session_state.user_text:
    user_text_feat = roberta.extract_features(roberta.encode(st.session_state.user_text.lower().strip()))
    text_to_heatmap(user_text_feat)
else:
    text_to_heatmap(precomp_text_feat)