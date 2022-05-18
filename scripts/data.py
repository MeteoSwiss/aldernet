# Standard library
import os

# Third-party
import cfgrib
import xarray as xr

variables = ["longitude", "latitude", "ALNUtune"]

os.system("rm -r /scratch/sadamov/aldernet/baseline")

ds = cfgrib.open_dataset(
    "/store/s83/osm/KENDA-1/ANA22/det/laf2022020900",
    backend_kwargs={"indexpath": "", "errors": "ignore",
                    'filter_by_keys': {'typeOfLevel': 'surface'}}
)
# ds = ds.drop_vars(("valid_time", "step", "surface")).expand_dims({"time": 1})
ds = ds[variables].expand_dims({"time": 1})
ds.to_zarr("/scratch/sadamov/aldernet/baseline")

path = "/store/s83/osm/KENDA-1/ANA22/det"
files = os.listdir(path)
files.sort()

# files_red = files[40 * 24:110 * 24]
files_red = files[40 * 24:41 * 24]

for file in files_red:
    ds = cfgrib.open_dataset(
        "/store/s83/osm/KENDA-1/ANA22/det/" + file,
        backend_kwargs={"indexpath": "", "errors": "ignore",
                        'filter_by_keys': {'typeOfLevel': 'surface'}}
    )
    ds = ds[variables].expand_dims({"time": 1})
    ds.to_zarr("/scratch/sadamov/aldernet/baseline", mode="a", append_dim="time")

# ds_final = xr.open_zarr("/scratch/sadamov/aldernet/baseline")
