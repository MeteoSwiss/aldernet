"""Train a cGAN and based on COSMO-1e input data."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator network

# Standard library
import datetime
import os
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import numpy as np
import tensorflow as tf
import xarray as xr

# First-party
from training_utils import experiment_path
from training_utils import generator
from training_utils import tf_setup
from training_utils import train_gan

os.chdir("/users/sadamov/PyProjects/aldernet/")

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

# Create Datasets
batch_size = 40

data = xr.open_zarr("/scratch/sadamov/aldernet/data").set_coords(
    ("longitude", "latitude", "time", "step", "valid_time")
)
data_reduced = data.isel(y=slice(450, 514), x=slice(500, 628))
pairs = np.load("data/cevio/pairs.npy").astype(int)

weather = np.load("data/weather.npy")
weather = np.delete(weather, [4, 7], axis=3)

# images tensor with 1224 samples
images_a = tf.convert_to_tensor(np.load("data/images_a.npy"))
images_b = tf.convert_to_tensor(np.load("data/images_b.npy"))
weather = tf.convert_to_tensor(weather)

# sys.stdout = open('outputfile', 'w')
# print(np.argwhere(np.isnan(weather)))

# for i in range(1):
#     data_reduced[list(data_reduced.keys())[i]].data.sum()

# np.save("images_a", data_reduced.ALNU.data[:, :, :, np.newaxis])
# np.save("images_b", data_reduced.CORY.data[:, :, :, np.newaxis])
# np.save("weather", data_reduced.drop_vars(("CORY", "ALNU")).to_array().to_numpy())

dataset_train = (
    tf.data.Dataset.from_tensor_slices((images_a, weather, images_b))
    .shuffle(40000)
    .batch(40)
    .prefetch(tf.data.AUTOTUNE)
)

# Model

height = images_a.shape[1]
width = images_a.shape[2]
weather_features = weather.shape[3]

generator = generator(height, width, weather_features)
# betas need to be floats, or checkpoint restoration fails
optimizer_gen = tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.0, beta_2=0.9)

with open(run_path + "/generator_summary.txt", "w") as handle:
    with redirect_stdout(handle):
        generator.summary()
tf.keras.utils.plot_model(
    generator, to_file=run_path + "/generator.png", show_shapes=True, dpi=96
)

# Train

train_gan(generator, optimizer_gen, dataset_train, run_path)
