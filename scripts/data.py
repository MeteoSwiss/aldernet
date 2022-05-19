# Standard library
import os

# Third-party
import cfgrib

variables_surface = {"typeOfLevel": "surface", "shortName": "ALNUtune"}
variables_3D = {"shortName": "ALNU"}

# os.system("rm -r /scratch/sadamov/aldernet/baseline")

# ds_surface = cfgrib.open_dataset(
#     "/store/s83/osm/KENDA-1/ANA22/det/laf2022020900",
#     backend_kwargs={"indexpath": "", "errors": "ignore",
#                     "filter_by_keys": variables_surface}
# )
# ds_surface = ds_surface.drop_vars(("valid_time", "step", "surface"))

# ds_3D = cfgrib.open_dataset(
#     "/store/s83/osm/KENDA-1/ANA22/det/laf2022020900",
#     backend_kwargs={"indexpath": "", "errors": "ignore",
#                     "filter_by_keys": variables_3D}
# )
# ds_3D = ds_3D.sel({"generalVerticalLayer": 80}).drop_vars(
#     ("valid_time", "step", "generalVerticalLayer"))
# ds_combined = ds_surface.expand_dims({"time": 1}).merge(ds_3D.expand_dims({"time": 1}))
# ds_combined.to_zarr("/scratch/sadamov/aldernet/baseline")
# # encoding={"time": {"dtype": "timedelta64[s]"}})

# path = "/store/s83/osm/KENDA-1/ANA22/det"
# files = os.listdir(path)
# files.sort()

# files_red = files[40 * 24:110 * 24]
files_red = files[39 * 24 + 1 : 940]

for file in files_red:
    ds_surface = cfgrib.open_dataset(
        "/store/s83/osm/KENDA-1/ANA22/det/" + file,
        backend_kwargs={
            "indexpath": "",
            "errors": "ignore",
            "filter_by_keys": variables_surface,
        },
    )
    ds_surface = ds_surface.drop_vars(("valid_time", "step", "surface"))
    ds_3D = cfgrib.open_dataset(
        "/store/s83/osm/KENDA-1/ANA22/det/" + file,
        backend_kwargs={
            "indexpath": "",
            "errors": "ignore",
            "filter_by_keys": variables_3D,
        },
    )
    ds_3D = ds_3D.sel({"generalVerticalLayer": 80}).drop_vars(
        ("valid_time", "step", "generalVerticalLayer")
    )
    ds_combined = ds_surface.expand_dims({"time": 1}).merge(
        ds_3D.expand_dims({"time": 1})
    )
    ds_combined.to_zarr(
        "/scratch/sadamov/aldernet/baseline", mode="a", append_dim="time"
    )
