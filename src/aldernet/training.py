"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator network

# Standard library
import datetime
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import mlflow
import numpy as np
from keras.utils import plot_model
from pyprojroot import here
from ray import init
from ray import shutdown
from ray import tune
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from tensorflow import random

# First-party
from aldernet.training_utils import compile_generator
from aldernet.training_utils import tf_setup
from aldernet.training_utils import train_model
from aldernet.training_utils import train_model_simple

# ---> DEFINE SETTINGS HERE <--- #
tune_with_ray = True
noise_dim = 0
add_weather = True
filter_time = 4344
# -------------------------------#

run_path = str(here()) + "/output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if tune_with_ray:
    Path(run_path + "/viz/valid").mkdir(parents=True, exist_ok=True)

path_data = "/scratch/sadamov/aldernet/npy/small/"

tf_setup()
random.set_seed(1)

# Profiling and Debugging
# tf.profiler.experimental.server.start(6009)
# tf.data.experimental.enable_debug_mode()

hazel_train = np.load(path_data + "hazel_train.npy")
hazel_valid = np.load(path_data + "hazel_valid.npy")
alder_train = np.load(path_data + "alder_train.npy")
alder_valid = np.load(path_data + "alder_valid.npy")
if add_weather:
    weather_train = np.load(path_data + "weather_train.npy")
    weather_valid = np.load(path_data + "weather_valid.npy")
else:
    weather_train = np.empty(shape=[0, 0, 0, 0])
    weather_valid = np.empty(shape=[0, 0, 0, 0])

if tune_with_ray:
    height = hazel_train.shape[1]
    width = hazel_train.shape[2]
    weather_features = weather_train.shape[3]

    generator = compile_generator(height, width, weather_features, noise_dim)

    with open(run_path + "/generator_summary.txt", "w") as handle:
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
            input_train=hazel_train[0:filter_time, :, :, :],
            target_train=alder_train[0:filter_time, :, :, :],
            weather_train=weather_train[0:filter_time, :, :, :],
            input_valid=hazel_valid[0:filter_time, :, :, :],
            target_valid=alder_valid[0:filter_time, :, :, :],
            weather_valid=weather_valid[0:filter_time, :, :, :],
            run_path=run_path,
            noise_dim=noise_dim,
        ),
        metric="Loss",
        num_samples=1,
        resources_per_trial={"gpu": 1},  # Choose approriate Device
        stop={"training_iteration": 5},
        config={
            # define search space here
            "learning_rate": tune.choice([0.0001]),
            "beta_1": tune.choice([0.85]),
            "beta_2": tune.choice([0.97]),
            "batch_size": tune.choice([40]),
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
    subprocess.run(rsync_cmd, shell=True)
else:
    train_model_simple(hazel_train, alder_train, hazel_valid, alder_valid, epochs=10)
