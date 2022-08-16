"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator network

# Standard library
import datetime
import os
from contextlib import redirect_stdout
from operator import concat
from pathlib import Path

# Third-party
import numpy as np
import ray
import tensorflow as tf
import xarray as xr
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback

ray.shutdown()
ray.init(
    runtime_env={
        "working_dir": "/users/sadamov/PyProjects/aldernet/",
        "excludes": ["data/", "run__/", ".git/"],
        #   "py_modules": [training_utils]
    }
)
os.chdir("/users/sadamov/PyProjects/aldernet/")

# First-party
# import training_utils
try:
    # First-party
    from training_utils import compile_generator
    from training_utils import experiment_path
    from training_utils import tf_setup
    from training_utils import train_model
except Exception:
    execfile("src/training_utils.py")

tf_setup(26000)
tf.random.set_seed(1)

run_path = (
    experiment_path + "/run__/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)

Path(run_path).mkdir(parents=True, exist_ok=True)


# Profiling and Debugging

# tf.profiler.experimental.server.start(6009)
# tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)

# data = xr.open_zarr("/scratch/sadamov/aldernet/data").set_coords(
#     ("longitude", "latitude", "time", "step", "valid_time")
# )
# sys.stdout = open('outputfile', 'w')
# print(np.argwhere(np.isnan(weather)))

# data_reduced = data.isel(y=slice(450, 514), x=slice(500, 628))
# data_reduced = data_reduced.interpolate_na(dim="x", method="linear", fill_value="extrapolate")

# del data

# images_a = np.log10(data_reduced.CORY.data[:, :, :, np.newaxis] + 1)
# images_b = np.log10(data_reduced.ALNU.data[:, :, :, np.newaxis] + 1)
# weather = data_reduced.drop_vars(
#     ("CORY", "ALNU")).to_array().transpose(
#         "time", "y", "x", "variable").to_numpy()
# # weather_test = np.delete(weather_test, [4, 7], axis=3)
# min_val = weather.min(axis=(0, 1, 2), keepdims=True)
# max_val = weather.max(axis=(0, 1, 2), keepdims=True)
# weather = (weather - min_val) / (max_val - min_val)

# del data_reduced

# np.save("images_a.npy", images_a)
# np.save("images_b.npy", images_b)
# np.save("weather.npy", weather)

weather = np.load("data/weather.npy")
images_a = 10 ** np.load("data/images_a.npy") - 1
images_b = 10 ** np.load("data/images_b.npy") - 1

# weather = weather[:, :, :, (2, 4, 22)]

# weather.sum(axis=(0, 1, 2), keepdims=True)

# images tensor with 1224 samples
# images_a = tf.convert_to_tensor(images_a)
# images_b = tf.convert_to_tensor(images_b)
# weather = tf.convert_to_tensor(weather)

# for i in range(1):
#     data_reduced[list(data_reduced.keys())[i]].data.sum()


dataset_train = {"images_a": images_a, "weather": weather, "images_b": images_b}

# dataset_train = (
#         tf.data.Dataset.from_tensor_slices((dataset_train["images_a"],
#                                             dataset_train["weather"],
#                                             dataset_train["images_b"]))
#         .shuffle(40000)
#         .batch(40)
#         .prefetch(tf.data.AUTOTUNE)
#     )

# dataset_train = (
#     tf.data.Dataset.from_tensor_slices((images_a, weather, images_b))
#     .shuffle(40000)
#     .batch(40)
#     .prefetch(tf.data.AUTOTUNE)
# )

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

config = {}
config["beta_1"] = 0.9
config["beta_2"] = 0.999
config["learning_rate"] = 0.00001

tune.run(
    tune.with_parameters(
        train_model, generator=generator, dataset_train=dataset_train, run_path=run_path
    ),
    metric="Loss",
    num_samples=8,
    stop={"training_iteration": 120},
    config={
        # define search space here
        "learning_rate": tune.choice([0.0000001, 0.0000005, 0.000001]),
        "beta_1": tune.choice([0.8, 0.85, 0.9]),
        "beta_2": tune.choice([0.95, 0.97, 0.999]),
        "batch_size": tune.choice([10, 20, 40]),
    },
    resources_per_trial={"gpu": 1},
    callbacks=[MLflowLoggerCallback(experiment_name="Aldernet", save_artifact=True)],
)

# train_model(config, generator, dataset_train, run_path)
