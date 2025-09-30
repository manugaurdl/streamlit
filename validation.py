import sys
import torch
import os
sys.path.append("/home/manugaur/mllm_inversion")

from src.training import losses, utils as train_utils, lora_siglip
from src.evaluation import pointing_eval
from src.dataloaders import utils as data_utils, datasets
import src.utils as utils
import yaml


config = yaml.safe_load(open("/home/manugaur/mllm_inversion/src/configs/text_cond_siglip.yaml", "r"))
config['data']['path']= '/workspace/manugaur/mllm_inversion/'
config['data']['mscoco'] = "/storage/datasets/coco/"
config['data']['visual_genome'] = "/storage/datasets/visual_genome_non_coco/"
config['data']['mapillary'] = "/storage/datasets/mapillary/images"
datasets_dict = {}
datasets_dict['train'] = datasets[config['dataset_class']]("train", config)
datasets_dict['val'] = datasets[config['dataset_class']]("val", config)

loaders = data_utils.get_loaders(datasets_dict, 0, 1, config)

def init_model(img_size, ckpt_dir, ckpt):
    from src.models import FlamingoCrossAttn
    from src.training import lora_siglip
    from src.dataloaders.utils import eval_transform

    encoder = FlamingoCrossAttn(
                visual_encoder="siglip_vitl_14",
                text_encoder ="roberta",
                img_res = img_size,
                cross_attn_layers =[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
                cross_attn_ffn_mult =2,
                )

    lora_siglip(
        encoder,
        rank = 8,
        last_n_blocks=6
        )
    trained_wts = torch.load(
        os.path.join(ckpt_dir, ckpt),
        map_location="cpu",
        weights_only=False,
    )
    trained_wts = trained_wts['model_state_dict']
    trained_wts = {k.replace('visual_encoder.siglip.visual.trunk', 'visual_encoder.trunk'): v for k,v in trained_wts.items() if 'visual_encoder.siglip.visual.trunk'}

    state_dict = {}
    for k,v in encoder.state_dict().items():
        if k in trained_wts: 
            state_dict[k] = trained_wts[k]
        else:
            state_dict[k] = v

    encoder.load_state_dict(state_dict)
    # print(f"no of trained weight : {len(trained_wts)}")

    return encoder.eval(), eval_transform(img_size)

img_res = 182
patch_size = 14
data_dir = "/storage/users/manugaur/mllm_inversion"
if not os.path.isdir(data_dir): 
    data_dir = "/workspace/manugaur/mllm_inversion"
# df = load_df(data_dir)
#args
ckpt_dir = os.path.join(data_dir, "checkpoints")
ckpt = "segmaskonlinegt_vitl14_2.4m_samples_182res_bsz42_nmyp22ei.pth"
device = torch.device('cuda')
model, transform = init_model(img_res, ckpt_dir, ckpt)
model = train_utils.init_model(config)
model.to(device)
criterion = losses.SoftCrossEntropyLoss()


@torch.no_grad()
def run_validation(model, loader, criterion, config, device):
    model.eval()
    val_stats = {}
    val_stats = pointing_eval(loader, device, model, criterion, val_stats)
    utils.update_save_metric(val_stats, config)
    torch.cuda.empty_cache()
    
    for k, v in val_stats.items():
        print(f"{k} : {v}")

run_validation(model, loaders['val'], criterion, config, device)