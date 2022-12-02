"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator network

# Standard library
import datetime
import socket
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import mlflow  # type: ignore
import xarray as xr
from keras.utils import plot_model  # type: ignore
from pyprojroot import here  # type: ignore
from ray import init
from ray import shutdown
from ray import tune
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from tensorflow import random  # type: ignore

# First-party
from aldernet.data.data_utils import Batcher
from aldernet.training_utils import compile_generator
from aldernet.training_utils import define_filters
from aldernet.training_utils import tf_setup
from aldernet.training_utils import train_model
from aldernet.training_utils import train_model_simple

# ---> DEFINE SETTINGS HERE <--- #
tune_with_ray = True
zoom = ""
noise_dim = 0
epochs = 3
shuffle = False
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
elif "nid" in hostname:
    data_train = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/aldernet/"
        + zoom
        + "/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/aldernet/"
        + zoom
        + "/data_valid.zarr"
    )

if tune_with_ray:
    height = data_train.CORY.shape[1]
    width = data_train.CORY.shape[2]
    if add_weather:
        weather_features = len(data_train.drop_vars(("ALNU", "CORY")).data_vars)
    else:
        weather_features = 0
    filters = define_filters(zoom)
    generator = compile_generator(height, width, weather_features, noise_dim, filters)

    with open(run_path + "/generator_summary.txt", "w", encoding="UTF-8") as handle:
        with redirect_stdout(handle):
            generator.summary()
    plot_model(generator, to_file=run_path + "/generator.png", show_shapes=True, dpi=96)

    # Train

    # Use hyperparameter search functionality by ray tune and log experiment
    shutdown()
    init(
        runtime_env={
            "working_dir": str(here()),
            "excludes": ["data/", ".git/", "images/"],
        }
    )

    tune.run(
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
        # metric="Loss",
        num_samples=1,
        scheduler=tune.schedulers.ASHAScheduler(
            time_attr="training_iteration",
            metric="Loss",
            mode="min",
            max_t=epochs,
            grace_period=3,
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
        checkpoint_score_attr="Loss",
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
else:
    batcher_train = Batcher(
        data_train, batch_size=32, add_weather=add_weather, shuffle=shuffle
    )
    batcher_valid = Batcher(
        data_valid, batch_size=32, add_weather=add_weather, shuffle=shuffle
    )
    train_model_simple(
        batcher_train, batcher_valid, epochs=epochs, add_weather=add_weather, conv=conv
    )
