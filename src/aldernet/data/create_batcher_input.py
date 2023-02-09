"""Import and preprocess the data."""

# pylint: disable=R0801

# Third-party
# import numpy as np
import xarray as xr
from pyprojroot import here  # type: ignore

# First-party
from aldernet.data.data_utils import Params

# Data Import
# Import zarr archive for the years 2020-2022
data = xr.open_zarr("/scratch/sadamov/pyprojects_data/aldernet/data.zarr")

data_select = data[Params().combined]

# Reduce spatial extent for faster training
# data_zoom = data_select.isel(y=slice(450, 514), x=slice(500, 628))
data_zoom = data_select

# Check for missing data
# Impute missing data that can sporadically occur in COSMO - very few datapoints
# sys.stdout = open("outputfile", "w")
# print(np.argwhere(np.isnan(data_zoom.to_array().to_numpy())))
data_zoom = data_zoom.interpolate_na(dim="x", method="linear", fill_value="extrapolate")

# data_high.CORY.values = np.log10(data_high.CORY.values + 1)
# data_high.ALNU.values = np.log10(data_high.ALNU.values + 1)

data_train = data_zoom.sel(valid_time=slice("2020-01-01", "2021-12-31"))
data_valid = data_zoom.sel(valid_time=slice("2022-01-01", "2022-12-31"))

high_indices = (
    (data_train["CORY"].mean(dim=("x", "y")) > 5)
    | (data_train["ALNU"].mean(dim=("x", "y")) > 5)
    # & (data_train["CORY"].max(dim=("x", "y")) < 5000)
    # & (data_train["ALNU"].max(dim=("x", "y")) < 5000)
)
data_train = data_train.sel({"valid_time": data_train.valid_time[high_indices]})

center = data_train.mean()
scale = data_train.std()

with open(str(here()) + "/data/scaling.txt", "w", encoding="utf-8") as f:
    f.write(
        "center: "
        + str(center[Params().y].to_array().values[0])
        + "\n"
        + "scale: "
        + str(scale[Params().y].to_array().values[0])
    )

data_train_norm = (data_train - center) / scale
data_valid_norm = (data_valid - center) / scale

data_train_norm.chunk({"valid_time": 32, "y": 786, "x": 1170}).to_zarr(
    "/scratch/sadamov/pyprojects_data/aldernet/data_final/data_train.zarr"
)
data_valid_norm.chunk({"valid_time": 32, "y": 786, "x": 1170}).to_zarr(
    "/scratch/sadamov/pyprojects_data/aldernet/data_final/data_valid.zarr"
)
