"""Rechung Zarr archive into 100MB chunks."""

# Third-party
import numpy as np
import xarray as xr

# COMBINE ALL DATA

# data_2022 = xr.open_zarr("/scratch/sadamov/aldernet/data2022")
# data_2021 = xr.open_zarr("/scratch/sadamov/aldernet/data2021")
# data_2020 = xr.open_zarr("/scratch/sadamov/aldernet/data2020")

# data_2020.to_zarr("/scratch/sadamov/aldernet/data")
# data_2021.to_zarr("/scratch/sadamov/aldernet/data", mode="a", append_dim="valid_time")
# data_2022.to_zarr("/scratch/sadamov/aldernet/data", mode="a", append_dim="valid_time")

# After this I had to manually remove the data variable "nominalTop"

data = xr.open_zarr("/scratch/sadamov/aldernet/data")

data["cos_dayofyear"] = (
    xr.DataArray(data=np.cos(2 * np.pi * data["valid_time.dayofyear"] / 365.25))
    .expand_dims(
        {"y": data.dims.mapping["y"], "x": data.dims.mapping["x"]}, axis=(1, 2)
    )
    .assign_coords(
        coords=dict(
            valid_time=data.valid_time.data,
            latitude=(["y", "x"], data.latitude.data),
            longitude=(["y", "x"], data.longitude.data),
        )
    )
)

data["sin_dayofyear"] = (
    xr.DataArray(data=np.sin(2 * np.pi * data["valid_time.dayofyear"] / 365.25))
    .expand_dims(
        {"y": data.dims.mapping["y"], "x": data.dims.mapping["x"]}, axis=(1, 2)
    )
    .assign_coords(
        coords=dict(
            valid_time=data.valid_time.data,
            latitude=(["y", "x"], data.latitude.data),
            longitude=(["y", "x"], data.longitude.data),
        )
    )
)

data["cos_hourofday"] = (
    xr.DataArray(data=np.cos(2 * np.pi * data["valid_time.hour"] / 24.0))
    .expand_dims(
        {"y": data.dims.mapping["y"], "x": data.dims.mapping["x"]}, axis=(1, 2)
    )
    .assign_coords(
        coords=dict(
            valid_time=data.valid_time.data,
            latitude=(["y", "x"], data.latitude.data),
            longitude=(["y", "x"], data.longitude.data),
        )
    )
)

data["sin_hourofday"] = (
    xr.DataArray(data=np.sin(2 * np.pi * data["valid_time.hour"] / 24.0))
    .expand_dims(
        {"y": data.dims.mapping["y"], "x": data.dims.mapping["x"]}, axis=(1, 2)
    )
    .assign_coords(
        coords=dict(
            valid_time=data.valid_time.data,
            latitude=(["y", "x"], data.latitude.data),
            longitude=(["y", "x"], data.longitude.data),
        )
    )
)

new_fn = "/scratch/sadamov/aldernet/data.zarr"
for i, var in enumerate(data.data_vars):
    # clear zarr encoding
    data[var].encoding.clear()
    if i == 0:
        data[[var]].chunk("auto").persist().to_zarr(new_fn, mode="w")
    else:
        data[[var]].chunk("auto").persist().to_zarr(new_fn, mode="a")
