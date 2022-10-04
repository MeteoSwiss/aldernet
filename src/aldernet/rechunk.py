"""Rechung Zarr archive into 100MB chunks."""

# Third-party
import xarray as xr

data = xr.open_zarr("/scratch/sadamov/aldernet/data")
new_fn = "/scratch/sadamov/aldernet/data.zarr"
for i, var in enumerate(data.data_vars):
    # clear zarr encoding
    data[var].encoding.clear()
    if i == 0:
        data[[var]].chunk("auto").persist().to_zarr(new_fn, mode="w")
    else:
        data[[var]].chunk("auto").persist().to_zarr(new_fn, mode="a")
