"""Training a Diffusion Model with PyTorch."""

# Third-party
import torch
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch import Unet

model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(
    model, image_size=128, timesteps=1000, loss_type="l1"  # number of steps  # L1 or L2
)
training_images = torch.randn(8, 3, 128, 128)
loss = diffusion(training_images)
loss.backward()
sampled_images = diffusion.sample(batch_size=4)

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

diffusion = GaussianDiffusion(
    model, image_size=128, timesteps=1000, loss_type="l1"  # number of steps  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    "path/to/your/images",
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
)

trainer.train()
