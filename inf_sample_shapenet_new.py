from ebm_finetune import train_util, utils
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    Sampler_create_gaussian_diffusion,
)
import argparse
import torch
import os
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--sampler', type=str, default="mala",choices=["MALA", "HMC", "UHMC", "ULA","Rev_Diff"])
args = parser.parse_args()

guidance_scale = 8

num_samples = 64
batch_size = 16
num_batches = num_samples // batch_size

sample_respacing = '100'
outputs_dir = "sample_out_128"

os.makedirs(outputs_dir, exist_ok=True)

def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    options = model_and_diffusion_defaults()

    #128x128
    model_path1= args.ckpt_path # 
    options["noise_schedule"]= "squaredcos_cap_v2"
    options["learn_sigma"] = False
    options["use_fp16"] = False
    options["num_classes"] = None  # "4,"
    options["dataset"] = "shapenet"
    options["image_size"] =   128#  128 , 3 
    options["num_channels"] = 128 #192 
    options["num_res_blocks"] = 2 #2
    options["energy_mode"] = True
    options["dataset"] = 'shapenet'
    
    # gt_options = {
    #     'image_size': 128, 
    #     'num_channels': 128, 
    #     'num_res_blocks': 2, 
    #     'channel_mult': '', 
    #     'num_heads': 1, 
    #     'num_head_channels': 64, 
    #     'num_heads_upsample': -1, 
    #     'attention_resolutions': '32,16,8', 
    #     'dropout': 0.1, 
    #     'text_ctx': 128, 
    #     'xf_width': 512, 
    #     'xf_layers': 16, 
    #     'xf_heads': 8, 
    #     'xf_final_ln': True, 
    #     'xf_padding': True, 
    #     'diffusion_steps': 1000, 
    #     'noise_schedule': 'squaredcos_cap_v2', 
    #     'timestep_respacing': '', 
    #     'use_scale_shift_norm': False, 
    #     'resblock_updown': True, 
    #     'use_fp16': False, 
    #     'cache_text_emb': False, 
    #     'inpaint': False, 
    #     'super_res': False, 
    #     'raw_unet': True, 
    #     'learn_sigma': False, 
    #     'use_kl': False, 
    #     'rescale_timesteps': False, 
    #     'rescale_learned_sigmas': False,
    #     'num_classes': None, 
    #     'dataset': 'shapenet', 
    #     'energy_mode': True}
    
    # for k, v in options.items():
        # if k not in gt_options:
        #     print((k, v), ' not in gt options!')
        #     print('-'*20)
        # elif gt_options[k] != v:
        #     print('keyword ', k, ' not match!')
        #     print('mine:' , v)
        #     print('gt: ', gt_options[k])

    # base_timestep_respacing = '100'
    
    weights = torch.load(model_path1, map_location="cpu")
    model,_ = create_model_and_diffusion(**options)
    # model.load_state_dict(weights, strict=True)
    # model.num_classes = None
    model.to(device)
    
    sampler = utils.energy_sample
    
    all_samples = []
    for _ in range(num_batches):
        samples =sampler(
            uncond = True,
            model=model,
            options=options,
            batch_size=batch_size,
            guidance_scale=guidance_scale, 
            device=device,
            prediction_respacing=sample_respacing,
        ).detach()
        
        all_samples.append(samples)
        
    all_samples = torch.cat(all_samples)
    all_samples = (all_samples + 1) / 2
    
    image_grid = make_grid(all_samples, nrow=8)
    sample_save_path = os.path.join(outputs_dir, f"sample.png")
    save_image(image_grid, sample_save_path)
    # train_util.pred_to_pil(samples).save(sample_save_path)
    
if __name__ == '__main__':
    main()