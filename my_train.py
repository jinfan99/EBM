import torch
from torchvision.utils import save_image, make_grid
import os 
import os.path as osp
from glob import glob
import argparse
import random 
import numpy as np 

from ebm_finetune.utils import load_model

from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults
)

from ebm_distill.distill import run_ebm_finetune_iter, EarlyStopper


def run_distillation(
    batch_size=1,
    learning_rate=1e-5,
    resume_ckpt="",
    checkpoints_dir="",
    log_frequency=100,
    project_name='ebm_distill',
    num_iters=100000,
    sample_bs=1,
    sample_gs=8.0,
    outputs_dir = './outputs_distill',
    num_classes = "",
    buffer = False,
    # learn_sigma = args.learn_sigma,
    noise_schedule = 'squaredcos_cap_v2',
    uncond = True,
    energy_mode = False,
    shapenet=True,
    # shapenet_train_only=args.shapenet_train_only
    teacher_ckpt_path="/home/zjf/repo/ebm_new/model-45x500.pt",
    verbose=False
):
    device = torch.device("cuda")
    
    model, diffusion, options = load_model(
        is_master=True,
        energy_mode=False,
        noise_schedule=noise_schedule,
        learn_sigma=False,
        num_classes=num_classes,
        model_type="base",
        shapenet=shapenet
    )

    model.to(device)
    
    teacher_model = load_teacher_model(teacher_ckpt_path).to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.0)
    # early_stopper = EarlyStopper(patience=3, min_delta=10)

    timestep_stopper = EarlyStopper(patience=10, min_delta=500, verbose=True, name='timestep stopper')
    
    curr_timestep = len(diffusion.betas) - 1 
    timesteps = curr_timestep * torch.ones((batch_size,), device=device)
    
    # print('betas:', diffusion.betas)
    batch_idx = 1
    iter_idx = 1
    
    while curr_timestep != 0:
        print('current timestep:' , curr_timestep)
        print('current batch: ', batch_idx)
        print('current iteration: ', iter_idx)
        timesteps = curr_timestep * torch.ones((batch_size,), device=device, dtype=torch.int)
        curr_energy = None
        
        while not timestep_stopper.early_stop(curr_energy):
            batch_stopper = EarlyStopper(patience=3, min_delta=5, verbose=verbose, name='batch stopper')
            # print('?????')
            curr_energy, iter_idx = run_ebm_finetune_iter(
                            uncond = uncond,
                            device = device,
                            model=model,
                            diffusion=diffusion,
                            timesteps=timesteps,
                            options=options,
                            optimizer=optimizer,
                            sample_bs=sample_bs,
                            sample_gs=sample_gs,
                            checkpoints_dir=checkpoints_dir,
                            outputs_dir=outputs_dir,
                            log_frequency=log_frequency,
                            batch_idx=batch_idx,
                            iter_idx=iter_idx,
                            gradient_accumualation_steps=1,
                            energy_mode= energy_mode,
                            shapenet=shapenet,
                            teacher_model=teacher_model,
                            early_stopper=batch_stopper,
                            batch_size=batch_size,
                        )
            
            batch_idx += 1
            
        curr_timestep -= 1
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )

    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./distill_checkpoints/"
    )
    
    parser.add_argument(
        "--teacher_ckpt_path", "-teacher_ckpt", type=str, default="/home/zjf/repo/ebm_new/model-45x500.pt"
    )
    
    parser.add_argument(
        "--outputs_dir",  type=str, default="./distill_outs/"
    )
    parser.add_argument(
        "--num_classes",  type=str, default=""
    )
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    # parser.add_argument("--image_size", "-freq", type=int, default=128)
    parser.add_argument("--project_name", "-name", type=str, default="shapenet-distill")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    # parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer", action="store_true")
    parser.add_argument("--num_iters", "-iters", type=int, default=1000000)

    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )

    parser.add_argument(
        "--energy_mode",
        action="store_true",
        help="Energy_mode",
    )
    
    parser.add_argument(
        "--shapenet",
        action="store_true",
        help="use shapenet-car dataset",
    )
    
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )

    parser.add_argument(
        "--buffer_size",type=int, default=1000, help="Buffer(Replay) Size"
    )
    parser.add_argument(
        "--noise_schedule",  type=str, default="squaredcos_cap_v2",choices=["squaredcos_cap_v2","linear"]
    )
    
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=4.0,
        help="Guidance scale used during model eval, not training.",
    )
   
    args = parser.parse_args()

    return args


def load_teacher_model(ckpt_path):
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    options = model_and_diffusion_defaults()

    #128x128
    model_path1= ckpt_path # 
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
    model.load_state_dict(weights, strict=True)
    # model.num_classes = None
    model.to(device)
    
    return model

def setup_seed(seed):
    # print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # th.backends.cudnn.deterministic = True
    
def main():
    args = parse_args()
    seeds= args.seed
    print("Setting the Seed to ", seeds)
    setup_seed(seed=seeds)
    
    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")
        
    isExist = os.path.exists(args.outputs_dir)
    if not isExist:
        os.makedirs(args.outputs_dir, exist_ok=True)

        
    # NUM_ITERATIONS = 100000
    # CKPT_PATH = "/home/zjf/re/EBM_new/checkpoints/Energy_object_chkpt_shapenet_alldata/model-50x2000.pt"
    # teacher_model = load_teacher_model(CKPT_PATH)
    
    
    # for n_iter in range(NUM_ITERATIONS):
    #     pass
    
    run_distillation(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        log_frequency=args.log_frequency,
        project_name=args.project_name,
        num_iters=args.num_iters,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        outputs_dir =args.outputs_dir,
        num_classes = args.num_classes,
        buffer = args.buffer,
        # learn_sigma = args.learn_sigma,
        noise_schedule = args.noise_schedule,
        uncond = args.uncond,
        energy_mode = args.energy_mode,
        shapenet=args.shapenet,
        # shapenet_train_only=args.shapenet_train_only
        teacher_ckpt_path=args.teacher_ckpt_path
    )
    


if __name__ == '__main__':
    main()