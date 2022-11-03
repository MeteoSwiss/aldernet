"""Import and preprocess the data."""

# Third-party
import numpy as np
import xarray as xr

# Data Import
# Import zarr archive for the years 2020-2022
data = xr.open_zarr("/scratch/sadamov/aldernet/data.zarr")

# Check for missing data (should not occur in model fields)
# sys.stdout = open("outputfile", "w")
# print(np.argwhere(np.isnan(data_zoom.to_array().to_numpy())))
# Impute missing data that can sporadically occur in COSMO-output (very few datapoints)
# data = data.interpolate_na(
#     dim="x", method="linear", fill_value="extrapolate"
# )

# Reduce spatial extent for faster training
data_zoom = data.isel(y=slice(450, 514), x=slice(500, 628))

high_indices = (
    (
        (data_zoom["CORY"].mean(dim=("x", "y")) > 30)
        | (data_zoom["ALNU"].mean(dim=("x", "y")) > 30)
    )
    & (data_zoom["CORY"].max(dim=("x", "y")) < 5000)
    & (data_zoom["ALNU"].max(dim=("x", "y")) < 5000)
)
data_high = data_zoom.sel({"valid_time": data_zoom.valid_time[high_indices]})

data_high.CORY.values = np.log10(data_high.CORY.values + 1)
data_high.ALNU.values = np.log10(data_high.ALNU.values + 1)

data_train = data_high.sel(valid_time=slice("2020-01-01", "2021-12-31")).transpose(
    "valid_time", "y", "x"
)
data_valid = data_high.sel(valid_time=slice("2022-01-01", "2022-12-31")).transpose(
    "valid_time", "y", "x"
)

center = data_train.mean()
scale = data_train.std()

data_train_norm = (data_train - center) / scale
data_valid_norm = (data_valid - center) / scale

# Pollen input field for Hazel
hazel_train = data_train_norm.CORY.values[:, :, :, np.newaxis]
hazel_valid = data_valid_norm.CORY.values[:, :, :, np.newaxis]
# Pollen output field for Alder
alder_train = data_train_norm.ALNU.values[:, :, :, np.newaxis]
alder_valid = data_valid_norm.ALNU.values[:, :, :, np.newaxis]
# Selection of additional weather parameters on ground level (please select the ones you like)
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
weather_train = (
    data_train_norm.drop_vars(("CORY", "ALNU"))[weather_params]
    .to_array()
    .transpose("valid_time", "y", "x", "variable")
    .to_numpy()
)
weather_valid = (
    data_valid_norm.drop_vars(("CORY", "ALNU"))[weather_params]
    .to_array()
    .transpose("valid_time", "y", "x", "variable")
    .to_numpy()
)

np.save("/scratch/sadamov/aldernet/npy/small/hazel_train.npy", hazel_train)
np.save("/scratch/sadamov/aldernet/npy/small/hazel_valid.npy", hazel_valid)
np.save("/scratch/sadamov/aldernet/npy/small/alder_train.npy", alder_train)
np.save("/scratch/sadamov/aldernet/npy/small/alder_valid.npy", alder_valid)
np.save("/scratch/sadamov/aldernet/npy/small/weather_train.npy", weather_train)
np.save("/scratch/sadamov/aldernet/npy/small/weather_valid.npy", weather_valid)
