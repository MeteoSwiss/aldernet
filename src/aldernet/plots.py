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
INPUT = "ml"
health_impact_groups = True
output_path = os.path.join(str(here()), "output/gif/")
output_name = "model_" + INPUT + "_2022_100-150_categ.gif"
scratch = os.environ["SCRATCH"]

# %%
if INPUT == "cosmo":
    ds = xr.open_zarr(scratch + "/pyprojects_data/aldernet/data_final/data_valid.zarr")
else:
    ds = xr.open_dataset(str(here()) + "/data/pollen_ml.nc")

# %%
if INPUT == "cosmo":
    with open(str(here()) + "/data/scaling.txt", "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
        center = float(lines[0].split(": ")[1])
        scale = float(lines[1].split(": ")[1])

        ds["ALNU"].values = np.maximum(0, ds["ALNU"].values * scale + center)

# %%
if health_impact_groups:
    ds["ALNU"].values = np.digitize(ds["ALNU"].values, bins=[1, 10, 70, 250])


# %%
# for valid_time in range(999, len(ds.ALNU.valid_time)):
for index in range(100, 150):
    plot1 = ds.psy.plot.mapplot(
        name="ALNU",
        valid_time=index,
        sort=["valid_time"],
        title="Alder Pollen in the Alpine Region on "
        + str(ds.valid_time[index].values)[0:13],
        titlesize=14,
        lakes=True,
        borders=True,
        rivers=True,
        grid_color="white",
        cticksize=8,
        clabel="Health Impact Categories",
        grid_labelsize=8,
        projection="robin",
        cmap="viridis",
    )

    if health_impact_groups:
        colorbar_ticks = [0.5, 1.5, 2.5, 3.5, 4.5]
        colorbar_bounds = list(range(0, 6, 1))
        colorbar_tickslabel = ["nothing", "weak", "medium", "strong", "very strong"]
        plot1.update(
            bounds=colorbar_bounds,
            cticks=colorbar_ticks,
            cticklabels=colorbar_tickslabel,
        )
    else:
        colorbar_bounds = list(range(0, 1001, 1))
        colorbar_ticks = [float(i) for i in colorbar_bounds]
        plot1.update(bounds=colorbar_bounds, cticks=colorbar_ticks)

    plt.ioff()
    plot1.export(output_path + "/map" + str(index).zfill(4) + ".png")
    plt.ion()
    plt.close()


# %%
frames = [Image.open(image) for image in sorted(glob.glob(f"{output_path}/*.png"))]
frame_one = frames[0]
frame_one.save(
    output_path + output_name,
    format="GIF",
    append_images=frames,
    save_all=True,
    duration=200,
    loop=0,
)
