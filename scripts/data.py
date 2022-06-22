# Standard library
import glob
import os

# Third-party
import cfgrib

# Remove existing zarr archive
os.system("rm -r /scratch/sadamov/aldernet/data")

# Variables to be extracted from surface-level GRIB file
# Unfortunately, all variables have to be read first (metadata only - lazily),
# as filtered import only supports one shortName at a time
var_selection = [
    "ALB_DIF",
    "ALB_RAD",
    "ALNUfr",
    "CLCT",
    "CORYctsum",
    "CORYfe",
    "CORYhcem",
    "CORYreso",
    "CORYress",
    "CORYrprec",
    "CORYsaisa",
    "CORYsaisn",
    "CORYsdes",
    "CORYtthre",
    "CORYtthrs",
    "CORYtune",
    "DPSDT",
    "DURSUN",
    "EMIS_RAD",
    "FIS",
    "FOR_D",
    "FOR_E",
    "FR_LAND",
    "HPBL",
    "HSURF",
    "LAI",
    "PLCOV",
    "PS",
    "ROOTDP",
    "RSMIN",
    "SKC",
    "SKYVIEW",
    "SLO_ANG",
    "SLO_ASP",
    "SOILTYP",
    "TCH",
    "TCM",
    "TQC",
    "TQV",
    "TWATER",
    "T_G",
    "W_I",
]

var_selection.sort()

ds_surface = cfgrib.open_dataset(
    "/scratch/sadamov/aldernet/000/laf2022020900",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "surface"},
    },
    encode_cf=("time", "geography", "vertical"),
)
ds_surface = ds_surface[var_selection]

ds_cory = cfgrib.open_dataset(
    "/scratch/sadamov/aldernet/000/laf2022020900",
    backend_kwargs={
        "filter_by_keys": {"shortName": "CORY"},
    },
    encode_cf=("time", "geography", "vertical"),
)
ds_cory = ds_cory.sel({"generalVerticalLayer": 80}).drop_vars(("generalVerticalLayer"))
ds_combined = ds_surface.expand_dims({"time": 1}).merge(
    ds_cory.expand_dims({"time": 1})
)

ds_alnu = cfgrib.open_dataset(
    "/scratch/sadamov/aldernet/000/laf2022020900",
    backend_kwargs={
        "filter_by_keys": {"shortName": "ALNU"},
    },
    encode_cf=("time", "geography", "vertical"),
)
ds_alnu = ds_alnu.sel({"generalVerticalLayer": 80}).drop_vars(("generalVerticalLayer"))
ds_combined = ds_combined.merge(ds_alnu.expand_dims({"time": 1}))
# ds_combined = ds_combined.assign_coords(member=1).expand_dims({"member": 1})

ds_combined.to_zarr("/scratch/sadamov/aldernet/data")

path = "/scratch/sadamov/aldernet/000/"
files = list(set(glob.glob(path + "*")) - set(glob.glob(path + "*.*")))
files.sort()

files_red = files[39 * 24 + 1 : 90 * 24]

for file in files_red:
    ds_surface = cfgrib.open_dataset(
        file,
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "surface"},
        },
        encode_cf=("time", "geography", "vertical"),
    )
    ds_surface = ds_surface[var_selection]

    ds_cory = cfgrib.open_dataset(
        file,
        backend_kwargs={
            "filter_by_keys": {"shortName": "CORY"},
        },
        encode_cf=("time", "geography", "vertical"),
    )
    ds_cory = ds_cory.sel({"generalVerticalLayer": 80}).drop_vars(
        ("generalVerticalLayer")
    )
    ds_combined = ds_surface.expand_dims({"time": 1}).merge(
        ds_cory.expand_dims({"time": 1})
    )
    ds_alnu = cfgrib.open_dataset(
        file,
        backend_kwargs={
            "filter_by_keys": {"shortName": "ALNU"},
        },
        encode_cf=("time", "geography", "vertical"),
    )
    ds_alnu = ds_alnu.sel({"generalVerticalLayer": 80}).drop_vars(
        ("generalVerticalLayer")
    )
    ds_combined = ds_combined.merge(ds_alnu.expand_dims({"time": 1}))

    ds_combined.to_zarr("/scratch/sadamov/aldernet/data", mode="a", append_dim="time")
