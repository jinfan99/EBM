import torch
from torchvision.utils import save_image, make_grid
import os 
import os.path as osp
from glob import glob


def load_teacher_model(ckpt_path):
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    options = model_and_diffusion_defaults()

    #128x128
    model_path1= args.ckpt_path # 
    options["noise_schedule"]= "linear"
    options["learn_sigma"] = False
    options["use_fp16"] = False
    options["num_classes"] = None  # "4,"
    options["dataset"] = "shapenet"
    options["image_size"] =   128#  128 , 3 
    options["num_channels"] = 128 #192 
    options["num_res_blocks"] = 2 #2
    options["energy_mode"] = True

    # base_timestep_respacing = '100'
    
    weights = torch.load(model_path1, map_location="cpu")
    model,_ = create_model_and_diffusion(**options)
    # model.load_state_dict(weights, strict=True)
    # model.num_classes = None
    model.to(device)
    
def main():
    pass
