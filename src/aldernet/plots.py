"""Create a Season GIF Animation."""

# pylint: disable=W0611

# %%
# Standard library
import glob
import os

# Third-party
import cfgrib  # type:ignore # noqa: F401
import iconarray  # type:ignore # noqa: F401
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import psyplot.project as psy  # type:ignore # noqa: F401
import xarray as xr
import zarr  # type:ignore # noqa: F401
from PIL import Image
from pyprojroot import here  # type:ignore

# %%
output_path = os.path.join(str(here()), "output/gif/")

# %%
ds = xr.open_zarr(
    "/scratch/sadamov/pyprojects_data/aldernet/no_threshold/data_valid.zarr"
)
# ds = xr.open_dataset(str(here()) + "/data/pollen_ml.nc")

# %%
with open(str(here()) + "/data/scaling.txt", "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]
    center = float(lines[0].split(": ")[1])
    scale = float(lines[1].split(": ")[1])

ds["ALNU"].values = np.maximum(0, ds["ALNU"].values * scale + center)

# %%
for valid_time in range(999, len(ds.ALNU.valid_time)):
    plot1 = ds.psy.plot.mapplot(
        name="ALNU",
        valid_time=valid_time,
        sort=["valid_time"],
        title="Alder Pollen in the Alps",
        titlesize=15,
        lakes=True,
        borders=True,
        rivers=True,
        grid_color="white",
        cticksize=8,
        grid_labelsize=8,
        projection="robin",
        cmap="RdBu_r",
    )

    colorbar = list(range(0, 1001, 50))
    plot1.update(bounds=colorbar, cticks=colorbar)

    plt.ioff()
    plot1.export(output_path + "/map" + str(valid_time).zfill(4) + ".png")
    plt.ion()
    plt.close()


# %%
frames = [Image.open(image) for image in sorted(glob.glob(f"{output_path}/*.png"))]
frame_one = frames[0]
frame_one.save(
    output_path + "/my_awesome.gif",
    format="GIF",
    append_images=frames,
    save_all=True,
    duration=200,
    loop=0,
)
