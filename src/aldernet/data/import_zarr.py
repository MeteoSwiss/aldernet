"""Import and preprocess the data."""

# Third-party
import numpy as np
import xarray as xr
from pyprojroot import here

# First-party
from aldernet.training_utils import normalize_field

# Data Import
# Import zarr archive for the years 2020-2022
data = xr.open_zarr("/scratch/sadamov/aldernet/data.zarr")
# Reduce spatial extent for faster training
data_zoom = data.isel(y=slice(450, 514), x=slice(500, 628))
# sys.stdout = open("outputfile", "w")
# print(np.argwhere(np.isnan(data_zoom.to_array().to_numpy())))
# Impute missing data that can sporadically occur in COSMO-output (very few datapoints)
data_zoom = data_zoom.chunk(dict(x=-1)).interpolate_na(
    dim="x", method="linear", fill_value="extrapolate"
)

data_train = data_zoom.sel(valid_time=slice("2020-01-01", "2021-12-31")).transpose(
    "valid_time", "y", "x"
)
data_valid = data_zoom.sel(valid_time=slice("2022-01-01", "2022-12-31")).transpose(
    "valid_time", "y", "x"
)

del data
del data_zoom

# Pollen input field for Hazel
hazel_train = data_train.CORY.values[:, :, :, np.newaxis]
hazel_valid = data_valid.CORY.values[:, :, :, np.newaxis]
# Pollen output field for Alder
alder_train = data_train.ALNU.values[:, :, :, np.newaxis]
alder_valid = data_valid.ALNU.values[:, :, :, np.newaxis]
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
    data_train.drop_vars(("CORY", "ALNU"))[weather_params]
    .to_array()
    .transpose("valid_time", "y", "x", "variable")
    .to_numpy()
)
weather_valid = (
    data_valid.drop_vars(("CORY", "ALNU"))[weather_params]
    .to_array()
    .transpose("valid_time", "y", "x", "variable")
    .to_numpy()
)

hazel_train = normalize_field(hazel_train, hazel_train)
hazel_valid = normalize_field(hazel_valid, hazel_train)
alder_train = normalize_field(alder_train, alder_train)
alder_valid = normalize_field(alder_valid, alder_train)
weather_train = normalize_field(weather_train, weather_train)
weather_valid = normalize_field(weather_valid, weather_train)

np.save(str(here()) + "/data/hazel_train.npy", hazel_train)
np.save(str(here()) + "/data/hazel_valid.npy", hazel_valid)
np.save(str(here()) + "/data/alder_train.npy", alder_train)
np.save(str(here()) + "/data/alder_valid.npy", alder_valid)
np.save(str(here()) + "/data/weather_train.npy", weather_train)
np.save(str(here()) + "/data/weather_valid.npy", weather_valid)
