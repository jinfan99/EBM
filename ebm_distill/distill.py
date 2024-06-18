from composable_diffusion.respace import SpacedDiffusion
import torch
from torchvision.utils import save_image
import os.path as osp

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, verbose=False, name='NO_NAME'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.loss_hist = []
        self.verbose = verbose
        self.name = name

    def early_stop(self, validation_loss=None):
        if validation_loss==None:
            return False
        
        self.loss_hist.append(validation_loss)
        
        # print('min delta: ', self.min_delta)
        # print('coming loss:', validation_loss)
        # print('min loss:', self.min_validation_loss)
        # print('history: ', self.loss_hist)
        
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.verbose:
                print('='*20)
                print('stopper name: ', self.name)
                print('min changed!')
                print('counter: ', self.counter)
                print('history: ', self.loss_hist)
                print('='*20)
        elif validation_loss < (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.verbose:
                print('='*20)
                print('stopper name: ', self.name)
                print('counter plus plus!')
                print('counter: ', self.counter)
                print('history: ', self.loss_hist)
                print('='*20)
            if self.counter >= self.patience:
                # if self.verbose:
                #     print('='*20)
                return True
            
        # print('='*20)
        return False
    
    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.loss_hist = []
        
def run_ebm_finetune_iter(
    uncond: bool,
    device,
    model,
    diffusion: SpacedDiffusion,
    timesteps: torch.Tensor,
    options: dict,
    optimizer: torch.optim.Optimizer,
    sample_bs: int,
    sample_gs: int,
    checkpoints_dir,
    outputs_dir,
    teacher_model,
    early_stopper: EarlyStopper,
    log_frequency=100,
    batch_idx=1,
    iter_idx=1,
    gradient_accumualation_steps=1,
    energy_mode= False,
    shapenet=True,
    batch_size=1
):
    model.to(device)
    model.train()
    
    teacher_model.to(device)
    # teacher_model.eval()
    
    # batch_size=options['batch_size']
    
    # timesteps = torch.randint(
    #     0, len(diffusion.betas) - 1, (batch_size,), device=device
    # )
    z_input = torch.randn((batch_size, 3, options['image_size'], options['image_size']), device=device)
    noise = torch.randn((batch_size, 3, options['image_size'], options['image_size']), device=device)
    
    gen_image = model(z_input, torch.tensor([0], device=device))   # fix timestep to be 0
    x_t = diffusion.q_sample(gen_image, timesteps, noise=noise).to(device)
    energy = teacher_model(x_t, timesteps, energy_only=True)

    energy.backward()
    optimizer.step()
    model.zero_grad()
        
    while not early_stopper.early_stop(energy.item()):
        gen_image = model(z_input, torch.tensor([0], device=device))   # fix timestep to be 0
        x_t = diffusion.q_sample(gen_image, timesteps, noise=noise).to(device)
        energy = teacher_model(x_t, timesteps, energy_only=True)

        energy.backward()
        optimizer.step()
        model.zero_grad()
        teacher_model.zero_grad()
        
        iter_idx += 1
    
    log = {"iter": batch_idx,"loss": energy.item() / gradient_accumualation_steps}
    
    if batch_idx == 1 or batch_idx % 10 == 0:
        save_image((gen_image + 1 ) / 2, osp.join(outputs_dir, f'{batch_idx}.png'))
        print(f'image sample {batch_idx}.png saved!')
        
    if batch_idx % 1000 == 0:
        torch.save(model, osp.join(checkpoints_dir, f'{batch_idx}.pth'))
        print(f'model {batch_idx}.pth saved!')
        
    print(f"loss: {energy.item():.4f}, batch: {batch_idx}")
    # print(f"Sampling from model at iteration {iter_idx}")
    
    # print('energy history: ', early_stopper.loss_hist)
    # early_stopper.reset()
    
    return energy.item(), iter_idx
    
    
