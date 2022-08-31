"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator network

# Standard library
import datetime
import os
from contextlib import redirect_stdout
from locale import normalize
from operator import concat
from pathlib import Path

# Third-party
import numpy as np
import ray
import tensorflow as tf
import xarray as xr
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"


os.chdir("/users/sadamov/PyProjects/aldernet/")
experiment_path = os.getcwd()

try:
    # First-party
    from training_utils import compile_generator
    from training_utils import normalize_field
    from training_utils import tf_setup
    from training_utils import train_model
except Exception:
    execfile("src/training_utils.py")

tf_setup()
tf.random.set_seed(1)

run_path = (
    experiment_path + "/run__/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)

Path(run_path + "/viz").mkdir(parents=True, exist_ok=True)

# Profiling and Debugging
# tf.profiler.experimental.server.start(6009)
# tf.data.experimental.enable_debug_mode()

update_input_data = False
if update_input_data:
    # Data Import
    data = xr.open_zarr("/scratch/sadamov/aldernet/data2022").set_coords(
        ("longitude", "latitude", "time", "step", "valid_time")
    )

    data_reduced = data.isel(y=slice(450, 514), x=slice(500, 628))

    sys.stdout = open("outputfile", "w")
    print(np.argwhere(np.isnan(data_reduced.to_array().to_numpy())))
    data_reduced = data_reduced.interpolate_na(
        dim="x", method="linear", fill_value="extrapolate"
    )

    del data

    images_a = data_reduced.CORY.data[:, :, :, np.newaxis]
    images_b = data_reduced.ALNU.data[:, :, :, np.newaxis]
    weather = (
        data_reduced.drop_vars(("CORY", "ALNU"))
        .to_array()
        .transpose("valid_time", "y", "x", "variable")
        .to_numpy()
    )

    images_a = normalize_field(images_a)
    images_b = normalize_field(images_b)
    weather = normalize_field(weather)

    del data_reduced

    np.save("data/images_a.npy", images_a)
    np.save("data/images_b.npy", images_b)
    np.save("data/weather.npy", weather)

weather = np.load("data/weather.npy")
weather = weather[:, :, :, (2, 4, 22)]
images_a = np.load("data/images_a.npy")
images_b = np.load("data/images_b.npy")

dataset_train = {"images_a": images_a, "weather": weather, "images_b": images_b}

# Model

height = images_a.shape[1]
width = images_a.shape[2]
weather_features = weather.shape[3]

generator = compile_generator(height, width, weather_features)

with open(run_path + "/generator_summary.txt", "w") as handle:
    with redirect_stdout(handle):
        generator.summary()
tf.keras.utils.plot_model(
    generator, to_file=run_path + "/generator.png", show_shapes=True, dpi=96
)

# Train

# Use hyperparameter search functionality by ray tune and log experiment
tune_with_ray = True

if tune_with_ray:
    ray.shutdown()
    ray.init(
        runtime_env={
            "working_dir": "/users/sadamov/PyProjects/aldernet/",
            "excludes": ["data/", "run__/", ".git/", "images/"],
            # "py_modules": ["/users/sadamov/PyProjects/aldernet/src/training_utils.py"]
        }
    )

    tune.run(
        tune.with_parameters(
            train_model,
            generator=generator,
            dataset_train=dataset_train,
            run_path=run_path,
            tune_with_ray=tune_with_ray,
        ),
        metric="Loss",
        num_samples=8,
        stop={"training_iteration": 4},
        config={
            # define search space here
            "learning_rate": tune.choice([0.00001, 0.00005, 0.0001]),
            "beta_1": tune.choice([0.8, 0.85, 0.9]),
            "beta_2": tune.choice([0.95, 0.97, 0.999]),
            "batch_size": tune.choice([10, 20, 40]),
        },
        resources_per_trial={"gpu": 1},
        callbacks=[
            MLflowLoggerCallback(experiment_name="Aldernet", save_artifact=True)
        ],
    )
else:
    config = {}
    config["beta_1"] = 0.85
    config["beta_2"] = 0.999
    config["learning_rate"] = 0.005
    config["batch_size"] = 20

    train_model(config, generator, dataset_train, run_path, tune_with_ray)
