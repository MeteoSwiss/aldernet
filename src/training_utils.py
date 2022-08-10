"""Train a cGAN with UNET Architecture.

To create realistic Images of Pollen Surface Concentration Maps.
"""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Standard library
import os
import time

# Third-party
import keras
import mlflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint
from tensorflow.linalg import matvec
from tensorflow.nn import l2_normalize

experiment_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

##########################


def tf_setup(memory_limit=8000):

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
        )


##########################


class SpectralNormalization(Constraint):
    def __init__(self, iterations=1):
        """Define class objects."""
        self.iterations = iterations
        self.u = None

    def __call__(self, w):
        output_neurons = w.shape[-1]
        W_ = tf.reshape(w, [-1, output_neurons])
        if self.u is None:
            self.u = tf.Variable(
                initial_value=tf.random_normal_initializer()(
                    shape=(output_neurons,), dtype="float32"
                ),
                trainable=False,
            )

        u_ = self.u
        v_ = None
        for _ in range(self.iterations):
            v_ = matvec(W_, u_)
            v_ = l2_normalize(v_)

            u_ = matvec(W_, v_, transpose_a=True)
            u_ = l2_normalize(u_)

        sigma = tf.tensordot(u_, matvec(W_, v_, transpose_a=True), axes=1)
        self.u.assign(u_)  # '=' produces an error in graph mode
        return w / sigma


##########################


def cbr(filters, name=None):

    block = keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def down(filters, name=None):

    block = keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def up(filters, name=None):

    block = keras.Sequential(name=name)
    block.add(
        layers.Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


##########################


filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]
noise_dim = 100
noise_channels = 128


def generator(height, width, weather_features):

    weather_input = keras.Input(
        shape=[height, width, weather_features], name="weather-input"
    )
    image_input = keras.Input(shape=[height, width, 1], name="image-input")
    inputs = layers.Concatenate(name="inputs-concat")([image_input, weather_input])

    block = cbr(filters[0], "pre-cbr-1")(inputs)

    u_skip_layers = [block]
    for ll in range(1, len(filters) // 2):

        block = down(filters[ll], "down_%s-down" % ll)(block)

        # Collect U-Net skip connections
        u_skip_layers.append(block)

    noise_input = keras.Input(shape=noise_dim, name="noise-input")
    height = block.shape[1]
    width = block.shape[2]
    noise = layers.Dense(
        height * width * noise_channels,
        # kernel_constraint=SpectralNormalization()
    )(noise_input)
    noise = layers.Reshape((height, width, -1))(noise)

    block = layers.Concatenate(name="add-noise")([block, noise])
    u_skip_layers.pop()

    for ll in range(len(filters) // 2, len(filters) - 1):

        block = up(filters[ll], "up_%s-up" % (len(filters) - ll - 1))(block)

        # Connect U-Net skip
        block = layers.Concatenate(name="up_%s-concatenate" % (len(filters) - ll - 1))(
            [block, u_skip_layers.pop()]
        )

    block = cbr(filters[-1], "post-cbr-1")(block)

    pollen = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="tanh",
        kernel_constraint=SpectralNormalization(),
        name="output",
    )(block)

    return tf.keras.Model(
        inputs=[noise_input, image_input, weather_input], outputs=pollen
    )


##########################


def write_png(image, path):
    image = (image + 1) * 127.5
    image = tf.cast(image, tf.uint8)
    png = tf.image.encode_png(image)
    tf.io.write_file(path, png)


bxe_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


##########################


filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]

# * https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# * Autodifferentiation vs. calculate training steps yourself
# * L1-Loss (Median) absolute value difference statt RMSE (Mean) als Loss - pixelwise
# * Calculate difference between image_b and generated ones for regression
# * Reduce output channels from 3 RGB to one pollen (+ height and width)
# * Weather input was simply zero mean and unit variance


@tf.function
def gan_step(
    generator,
    optimizer_gen,
    images_a,
    weather,
    images_b,
    step,
):

    noise = tf.random.normal([images_a.shape[0], noise_dim])
    with tf.GradientTape() as tape_gen:
        generated = generator([noise, images_a, weather])
        gen_loss = tf.math.reduce_sum(tf.math.abs(generated - images_b))
        gradients_gen = tape_gen.gradient(gen_loss, generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    print(gen_loss.numpy(), flush=True)
    mlflow.log_metric("Loss", gen_loss.numpy(), step=step.numpy())


###############################


def train_gan(generator, optimizer_gen, dataset_train, run_path):
    mlflow.set_experiment("Aldernet")
    mlflow.start_run()
    epoch = tf.Variable(1, dtype="int64")
    step = tf.Variable(1, dtype="int64")

    ckpt = tf.train.Checkpoint(
        epoch=epoch,
        step=step,
        generator=generator,
        optimizer_gen=optimizer_gen,
    )
    manager = tf.train.CheckpointManager(
        ckpt,
        directory=run_path + "/checkpoint",
        max_to_keep=2,
        keep_checkpoint_every_n_hours=4,
    )
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint), flush=True)
    else:
        print("Initializing from scratch.", flush=True)

    while True:
        start = time.time()
        for (
            images_a,
            weather,
            images_b,
        ) in dataset_train:

            print(epoch.numpy(), "-", step.numpy(), flush=True)

            gan_step(
                generator,
                optimizer_gen,
                images_a,
                weather,
                images_b,
                step,
            )

            noise_1 = tf.random.normal([images_a.shape[0], noise_dim])
            generated_1 = generator([noise_1, images_a, weather])
            viz = tf.concat([images_a[0], images_b[0], generated_1[0]], axis=1)

            write_png(
                viz,
                run_path
                + "/viz/"
                + str(epoch.numpy())
                + "-"
                + str(step.numpy())
                + ".png",
            )

            step.assign_add(1)

        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch.numpy(), time.time() - start
            ),
            flush=True,
        )
        epoch.assign_add(1)
        manager.save()
