"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Standard library
import datetime
import socket
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import mlflow  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import xarray as xr
from keras.utils import plot_model  # type: ignore
from pyprojroot import here  # type: ignore
from ray import init
from ray import shutdown
from ray import tune
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from tensorflow import random  # type: ignore

# First-party
from aldernet.data.data_utils import Batcher
from aldernet.data.data_utils import Stations
from aldernet.training_utils import compile_generator
from aldernet.training_utils import define_filters
from aldernet.training_utils import predict_season
from aldernet.training_utils import tf_setup
from aldernet.training_utils import train_model
from aldernet.training_utils import train_model_simple

# ---> DEFINE SETTINGS HERE <--- #
input_species = "CORY"
target_species = "ALNU"
retrain_model = True
tune_with_ray = True
zoom = ""
noise_dim = 100
epochs = 10
shuffle = True
add_weather = True
conv = False
members = 1
device = {"gpu": 4}
# -------------------------------#

if target_species == "ALNU":
    target_species_name = "Alnus"
else:
    target_species_name = ""

tf_setup()
random.set_seed(1)

run_path = str(here()) + "/output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if tune_with_ray and retrain_model:
    Path(run_path + "/viz/valid").mkdir(parents=True, exist_ok=True)

hostname = socket.gethostname()
if "tsa" in hostname:
    data_train = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/5_threshold"
        + zoom
        + "/data_final/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/5_threshold"
        + zoom
        + "/data_final/data_valid.zarr"
    )
elif "nid" in hostname:
    data_train = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/pyprojects_data/aldernet/"
        + zoom
        + "/data_final/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/pyprojects_data/aldernet/"
        + zoom
        + "/data_final/data_valid.zarr"
    )
else:
    data_train = xr.DataArray()
    data_valid = xr.DataArray()

if retrain_model:
    if tune_with_ray:
        height = data_train.CORY.shape[1]
        width = data_train.CORY.shape[2]
        if add_weather:
            weather_features = len(
                data_train.drop_vars((target_species, input_species)).data_vars
            )
        else:
            weather_features = 0
        filters = define_filters(zoom)
        generator = compile_generator(
            height, width, weather_features, noise_dim, filters
        )

        with open(run_path + "/generator_summary.txt", "w", encoding="UTF-8") as handle:
            with redirect_stdout(handle):
                generator.summary()
        plot_model(
            generator, to_file=run_path + "/generator.png", show_shapes=True, dpi=96
        )

        # Train

        # Use hyperparameter search functionality by ray tune and log experiment
        shutdown()
        init(
            runtime_env={
                "working_dir": str(here()),
                "excludes": ["data/", ".git/", "images/"],
            }
        )

        tuner = tune.run(
            tune.with_parameters(
                train_model,
                generator=generator,
                data_train=data_train,
                data_valid=data_valid,
                run_path=run_path,
                noise_dim=noise_dim,
                add_weather=add_weather,
                shuffle=shuffle,
            ),
            metric="Loss",
            mode="min",
            num_samples=members,
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                max_t=epochs,
                grace_period=2,
                reduction_factor=3,
            ),
            resources_per_trial=device,  # Choose appropriate Device
            config={
                # define search space here
                "learning_rate": tune.choice([0.001]),
                "beta_1": tune.choice([0.9, 0.95]),
                "beta_2": tune.choice([0.98, 0.99]),
                "mlflow": {
                    "experiment_name": "Aldernet",
                    "tracking_uri": mlflow.get_tracking_uri(),
                },
            },
            local_dir=run_path,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name="Aldernet",
                    tracking_uri=run_path + "/mlruns",
                    save_artifact=True,
                )
            ],
        )
        # rsync commands to merge the mlruns directories
        rsync_cmd = "rsync" + " -avzh " + run_path + "/mlruns" + " " + str(here())
        subprocess.run(rsync_cmd, shell=True, check=True)
        best_model = tuner.best_checkpoint.to_dict()["model"]
    else:
        batcher_train = Batcher(
            data_train, batch_size=32, add_weather=add_weather, shuffle=shuffle
        )
        batcher_valid = Batcher(
            data_valid, batch_size=32, add_weather=add_weather, shuffle=shuffle
        )
        simple_model = train_model_simple(
            batcher_train,
            batcher_valid,
            epochs=epochs,
            add_weather=add_weather,
            conv=conv,
        )
        best_model = simple_model
else:
    best_model = Checkpoint.from_directory(
        "/users/sadamov/pyprojects/aldernet/output/20221229_164219/"
        "train_model_2022-12-29_16-42-33/train_model_69560_00000_0_beta_1="
        "0.9500,beta_2=0.9800,"
        "learning_rate=0.0010_2022-12-29_16-42-52/checkpoint_000000"
    ).to_dict()["model"]

predictions = predict_season(best_model, data_valid, noise_dim, add_weather)

with open(str(here()) + "/data/scaling.txt", "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]
    center = float(lines[0].split(": ")[1])
    scale = float(lines[1].split(": ")[1])

data_valid[target_species] = (data_valid.dims, np.squeeze(predictions))
data_alnu_output = np.maximum(0, data_valid[target_species] * scale + center)

data_alnu_output.to_netcdf(str(here()) + "/data/pollen_ml.nc")  # type: ignore

station_values = data_alnu_output.values[  # type:ignore
    :, Stations().grids["grid_j"], Stations().grids["grid_i"]
]

data_pd = pd.concat(
    [
        pd.DataFrame(
            {"datetime": data_valid.valid_time.values, "taxon": target_species_name}
        ),
        pd.DataFrame(station_values),
    ],
    axis=1,
)

data_pd.columns = ["datetime", "taxon"] + Stations().name  # type: ignore

data_pd.to_csv(str(here()) + "/data/pollen_ml.atab", sep=",", index=False)

with subprocess.Popen(
    ["Rscript", "--vanilla", str(here()) + "/notebooks/rmd2html.R"], shell=False
):
    print("Creating html verification report.")
