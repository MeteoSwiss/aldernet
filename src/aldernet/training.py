"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=no-member,import-error,no-name-in-module,invalid-name
# pyright: reportUnboundVariable=false

# Standard library
import datetime
import socket
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import mlflow  # type: ignore
import numpy as np
import xarray as xr
from keras.utils import plot_model  # type: ignore
from pyprojroot import here  # type: ignore
from ray import init
from ray import shutdown
from ray import tune
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.air.checkpoint import Checkpoint
from tensorflow import random  # type: ignore

# First-party
from aldernet.data.data_utils import Batcher
from aldernet.training_utils import compile_generator
from aldernet.training_utils import define_filters
from aldernet.training_utils import tf_setup
from aldernet.training_utils import train_model
from aldernet.training_utils import train_model_simple

# ---> DEFINE SETTINGS HERE <--- #
retrain_model = False
tune_with_ray = True
zoom = "data_zoom"
noise_dim = 0
epochs = 3
shuffle = True
add_weather = False
conv = False
# -------------------------------#


tf_setup()
random.set_seed(1)

run_path = str(here()) + "/output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if tune_with_ray:
    Path(run_path + "/viz/valid").mkdir(parents=True, exist_ok=True)

hostname = socket.gethostname()
if "tsa" in hostname:
    data_train = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/" + zoom + "/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/" + zoom + "/data_valid.zarr"
    )
    data_season = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/no_threshold/"
        + zoom
        + "/data_valid.zarr"
    )
elif "nid" in hostname:
    data_train = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/pyprojects_data/aldernet/"
        + zoom
        + "/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/pyprojects_data/aldernet/"
        + zoom
        + "/data_valid.zarr"
    )
    data_season = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/"
        "pyprojects_data/aldernet/no_threshold/" + zoom + "/data_valid.zarr"
    )

if retrain_model:
    if tune_with_ray:
        height = data_train.CORY.shape[1]
        width = data_train.CORY.shape[2]
        if add_weather:
            weather_features = len(data_train.drop_vars(("ALNU", "CORY")).data_vars)
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
            num_samples=1,
            scheduler=tune.schedulers.ASHAScheduler(  # type: ignore
                time_attr="training_iteration",
                max_t=epochs,
                grace_period=1,
                reduction_factor=3,
            ),
            resources_per_trial={"gpu": 1},  # Choose appropriate Device
            # stop={"training_iteration": 2},
            config={
                # define search space here
                "learning_rate": tune.choice([0.0001]),
                "beta_1": tune.choice([0.85]),
                "beta_2": tune.choice([0.97]),
                "batch_size": tune.choice([10]),
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
        best_checkpoint = tuner.best_checkpoint
    else:
        batcher_train = Batcher(
            data_train, batch_size=32, add_weather=add_weather, shuffle=shuffle
        )
        batcher_valid = Batcher(
            data_valid, batch_size=32, add_weather=add_weather, shuffle=shuffle
        )
        train_model_simple(
            batcher_train,
            batcher_valid,
            epochs=epochs,
            add_weather=add_weather,
            conv=conv,
        )
else:
    best_checkpoint = Checkpoint.from_directory(
        "/users/sadamov/pyprojects/aldernet/output/"
        "20221220_132707/train_model_2022-12-20_13-27-23/"
        "train_model_a7ffb_00000_0_batch_size=10,beta_1=0.8500,beta_2=0.9700,"
        "learning_rate=0.0001_2022-12-20_13-27-25/checkpoint_000002"
    )

predictor = best_checkpoint.to_dict()["model"]

predictions = predictor.predict(
    Batcher(data_season, batch_size=32, add_weather=add_weather, shuffle=shuffle)
)

data_season["ALNU"] = (data_season.dims, np.squeeze(predictions))

# retrieve values at stations level and write into atab format
stations = {}

stations["i"] = [525, 511, 651, 677, 420, 472, 456, 603, 614, 570, 636, 479, 540, 590]
stations["j"] = [506, 444, 465, 429, 373, 463, 405, 365, 349, 455, 510, 451, 379, 485]

# stations["abbr"] = [
#     "PBS",
#     "PBE",
#     "PBU",
#     "PDS",
#     "PGE",
#     "PCF",
#     "PLS",
#     "PLO",
#     "PLU",
#     "PLZ",
#     "PMU",
#     "PNE",
#     "PVI",
#     "PZH",
# ]
