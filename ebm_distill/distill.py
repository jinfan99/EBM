from composable_diffusion.respace import SpacedDiffusion
import torch

def run_ebm_finetune_iter(
    uncond: bool,
    device,
    model,
    diffusion: SpacedDiffusion,
    options: dict,
    optimizer: torch.optim.Optimizer,
    sample_bs: int,
    sample_gs: int,
    checkpoints_dir,
    outputs_dir,
    teacher_model,
    early_stopper,
    log_frequency=100,
    iter_idx=0,
    gradient_accumualation_steps=1,
    energy_mode= False,
    shapenet=True,
):
    model.to(device)
    model.train()
    
    teacher_model.to(device)
    # teacher_model.eval()
    
    batch_size=options['batch_size']
    
    timesteps = torch.randint(
        0, len(diffusion.betas) - 1, (batch_size,), device=device
    )
    z_input = torch.randn((batch_size, 3, options['image_size'], options['image_size']), device=device)
    noise = torch.randn((batch_size, 3, options['image_size'], options['image_size']), device=device)
    
    energt_list = []
    while not energe_converged(energt_list):
        gen_image = model(z_input, 0)   # fix timestep to be 0
        x_t = diffusion.q_sample(gen_image, timesteps, noise=noise).to(device)
        energy = teacher_model(x_t, timesteps, energy_only=True)
        energt_list.append(energy)

        energy.backward()
        optimizer.step()
        model.zero_grad()
    
    log = {"iter": iter_idx,"loss": energy.item() / gradient_accumualation_steps}
    
    print(f"loss: {energy.item():.4f}")
    # print(f"Sampling from model at iteration {iter_idx}")
    
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.loss_hist = []

    def early_stop(self, validation_loss):
        self.loss_hist.append(validation_loss)
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.loss_hist = []