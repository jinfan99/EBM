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


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--sampler', type=str, default="mala",choices=["MALA", "HMC", "UHMC", "ULA","Rev_Diff"])
args = parser.parse_args()

guidance_scale = 4
batch_size = 1
sample_respacing = '100'
outputs_dir = "sample_out"

def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    options = model_and_diffusion_defaults()

    #128x128
    model_path1= args.ckpt_path # 
    options["noise_schedule"]= "linear"
    options["learn_sigma"] = False
    options["use_fp16"] = False
    options["num_classes"] = ""  # "4,"
    options["dataset"] = "clevr_norel"
    options["image_size"] =   128#  128 , 3 
    options["num_channels"] = 128 #192 
    options["num_res_blocks"] = 2 #2
    options["energy_mode"] = True

    # base_timestep_respacing = '100'
    
    weights = torch.load(model_path1, map_location="cpu")
    model,_ = create_model_and_diffusion(**options)
    model.load_state_dict(weights, strict=True)
    model.num_classes = None
    model.to(device)
    
    sampler = utils.energy_sample
    samples =sampler(
        uncond = True,
        model=model,
        options=options,
        batch_size=batch_size,
        guidance_scale=guidance_scale, 
        device=device,
        prediction_respacing=sample_respacing,
    ).detach()
    
    sample_save_path = os.path.join(outputs_dir, f"sample.png")
    train_util.pred_to_pil(samples).save(sample_save_path)
    
if __name__ == '__main__':
    main()