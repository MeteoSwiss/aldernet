"""Training a Diffusion Model with PyTorch."""

# Standard library
import socket

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Unet


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")


def save(img):
    npimg = img.numpy()
    plt.imsave(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")


zoom = ""
hostname = socket.gethostname()
if "tsa" in hostname:
    data_train = xr.open_zarr("/scratch/sadamov/aldernet/" + zoom + "/data_train.zarr")
    data_valid = xr.open_zarr("/scratch/sadamov/aldernet/" + zoom + "/data_valid.zarr")
elif "nid" in hostname:
    data_train = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/aldernet/"
        + zoom
        + "/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/aldernet/"
        + zoom
        + "/data_valid.zarr"
    )

data_img = (
    data_train.ALNU.expand_dims({"channel": 1})
    .isel(y=slice(400, 464), x=slice(400, 464), valid_time=slice(0, 16))
    .transpose("valid_time", "channel", "y", "x")
    .to_numpy()
)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1).cuda()
diffusion = GaussianDiffusion(
    model, image_size=64, timesteps=1000, loss_type="l1"
).cuda()

training_images = torch.from_numpy(data_img).cuda()
loss = diffusion(training_images).cuda()
loss.backward()

sampled_images = diffusion.sample(batch_size=4).cuda()

show(training_images.cpu()[0])
show(sampled_images.cpu()[0])

save(sampled_images.cpu()[0], "sample1.png")
save(sampled_images.cpu()[1], "sample2.png")
save(sampled_images.cpu()[2], "sample3.png")
save(sampled_images.cpu()[3], "sample4.png")

torch.cuda.empty_cache()
