"""Function scales and centers 4D numpy arrays (time, lat, lon, weather)."""

# pylint: disable-all

# Third-party
import numpy as np
import xarray as xr


def normalize_field(data):
    min_val = data.min(axis=(0, 1, 2), keepdims=True)
    max_val = data.max(axis=(0, 1, 2), keepdims=True)
    data = (data - min_val) / (max_val - min_val)
    return data


# Import zarr archive for the years 2020-2022
data = xr.open_zarr("/scratch/sadamov/aldernet/data")
# Reduce spatial extent for faster training
data_reduced = data.isel(
    valid_time=slice(0, 5568), y=slice(450, 514), x=slice(500, 628)
)
# Impute missing data that can sporadically occur in COSMO-output (very few datapoints)
data_reduced = data_reduced.chunk(dict(x=-1)).interpolate_na(
    dim="x", method="linear", fill_value="extrapolate"
)

# Pollen input field for Hazel
images_a = data_reduced.CORY.values[:, :, :, np.newaxis]
# Pollen output field for Alder
images_b = data_reduced.ALNU.values[:, :, :, np.newaxis]
# Selection of additional weather parameters on ground level
# Depending on the amount of weather fields this step takes several minutes to 1 hour.
weather_params = [
    "ALNUfr",
    "CORYctsum",
    "CORYfr",
    "DURSUN",
    "HPBL",
    "PLCOV",
    "T",
    "TWATER",
    "U",
    "V",
]
weather = (
    data_reduced.drop_vars(("CORY", "ALNU"))[weather_params]
    .to_array()
    .transpose("valid_time", "y", "x", "variable")
    .to_numpy()
)

# Make sure all fields live on the same scale [0,1]
images_a = normalize_field(images_a)
images_b = normalize_field(images_b)
weather = normalize_field(weather)
