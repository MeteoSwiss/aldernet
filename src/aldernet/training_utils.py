"""Train a cGAN with UNET Architecture.

To create realistic Images of Pollen Surface Concentration Maps.
"""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Standard library
import time
from pathlib import Path

# Third-party
import keras
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from keras import layers
from keras.constraints import Constraint
from ray import tune
from tensorflow.linalg import matvec
from tensorflow.nn import l2_normalize

##########################


def normalize_field(data, data_train):
    center = data_train.mean(axis=(0, 1, 2), keepdims=True)
    scale = data_train.std(axis=(0, 1, 2), keepdims=True)
    data = (data - center) / scale
    return data


##########################


def tf_setup():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


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
            # kernel_constraint=SpectralNormalization(),
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
            # kernel_constraint=SpectralNormalization(),
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
            # kernel_constraint=SpectralNormalization(),
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


##########################


filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]


def compile_generator(height, width, weather_features):

    weather_input = keras.Input(
        shape=[height, width, weather_features], name="weather_input"
    )
    image_input = keras.Input(shape=[height, width, 1], name="image_input")
    inputs = layers.Concatenate(name="inputs-concat")([image_input, weather_input])

    block = cbr(filters[0], "pre-cbr-1")(inputs)

    u_skip_layers = [block]
    for ll in range(1, len(filters) // 2):

        block = down(filters[ll], "down_%s-down" % ll)(block)

        # Collect U-Net skip connections
        u_skip_layers.append(block)

    height = block.shape[1]
    width = block.shape[2]
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
        # kernel_constraint=SpectralNormalization(),
        name="output",
    )(block)

    return tf.keras.Model(inputs=[image_input, weather_input], outputs=pollen)


##########################


def write_png(image, path, pretty):

    if pretty:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.1), dpi=150)
        ax1.imshow(image[0][:, :, 0], cmap="viridis")
        ax2.imshow(image[1][:, :, 0], cmap="viridis")
        ax3.imshow(image[2][:, :, 0], cmap="viridis")
        for ax in (ax1, ax2, ax3):
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        ax1.axes.set_title("Input")
        ax2.axes.set_title("Target")
        ax3.axes.set_title("Prediction")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(path)
        plt.close(fig)
    else:
        image = tf.concat(image, axis=1) * 10
        image = tf.cast(image, tf.uint8)
        png = tf.image.encode_png(image)
        tf.io.write_file(path, png)


##########################

bxe_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]

# * https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# * Autodifferentiation vs. calculate training steps yourself
# * L1-Loss (Median) absolute value difference statt RMSE (Mean) als Loss - pixelwise
# * Calculate difference between image_b and generated ones for regression
# * Reduce output channels from 3 RGB to one pollen (+ height and width)
# * Weather input was simply zero mean and unit variance


# @tf.function(jit_compile=True)
def gan_step(generator, optimizer_gen, input_train, target_train, weather_train):

    with tf.GradientTape() as tape_gen:
        generated = generator([input_train, weather_train])
        loss = tf.math.reduce_sum(tf.math.abs(generated - target_train))
        # loss = tf.math.reduce_mean(tf.math.squared_difference(generated, alder))
        gradients_gen = tape_gen.gradient(loss, generator.trainable_variables)

    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return loss


def train_model(
    config, generator, input_train, target_train, weather_train, run_path, tune_with_ray
):
    if tune_with_ray:
        mlflow.set_experiment("Aldernet")
        mlflow.set_tracking_uri("mlruns")
        tune_trial = tune.get_trial_name() + "/"
        Path(run_path + "/viz/" + tune_trial).mkdir(parents=True, exist_ok=True)
    else:
        tune_trial = ""
    epoch = tf.Variable(1, dtype="int64")
    step = tf.Variable(1, dtype="int64")

    dataset_train = (
        tf.data.Dataset.from_tensor_slices((input_train, target_train, weather_train))
        .shuffle(1000)
        .batch(config["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    # betas need to be floats, or checkpoint restoration fails
    optimizer_gen = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"],
        beta_1=config["beta_1"],
        beta_2=config["beta_2"],
        epsilon=1e-08,
    )

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
        for (hazel_train, alder_train, weather_train) in dataset_train:

            if not tune_with_ray:
                print(epoch.numpy(), "-", step.numpy(), flush=True)

            loss = gan_step(
                generator, optimizer_gen, hazel_train, alder_train, weather_train
            )

            generated_1 = generator([hazel_train, weather_train])
            viz = (hazel_train[0], alder_train[0], generated_1[0])

            write_png(
                viz,
                run_path
                + "/viz/"
                + tune_trial
                + str(epoch.numpy())
                + "-"
                + str(step.numpy())
                + ".png",
                pretty=True,
            )

            step.assign_add(1)

        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch.numpy(), time.time() - start
            ),
            flush=True,
        )
        if not tune_with_ray:
            manager.save()
        if tune_with_ray:
            tune.report(iterations=step, Loss=loss.numpy())
        epoch.assign_add(1)
    return {"Loss": loss}


def train_model_simple(
    training_input, training_target, validation_input, validation_target
):
    training_tensor = (
        tf.data.Dataset.from_tensor_slices((training_input, training_target))
        .shuffle(1000)
        .batch(40)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_tensor = (
        tf.data.Dataset.from_tensor_slices((validation_input))
        .batch(40)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = keras.models.Sequential()
    model.add(layers.Dense(1, input_shape=(64, 128, 1)))
    model.add(
        layers.Conv2D(
            filters=2,
            kernel_size=4,
            strides=1,
            padding="same",
            use_bias=False,
            # kernel_constraint=SpectralNormalization()
        )
    )
    model.add(layers.Dense(1, activation="linear"))
    model.summary()

    model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )

    model.fit(training_tensor, epochs=100, batch_size=40)
    predictions = model.predict(validation_tensor)
    for timestep in range(0, len(validation_input), 100):
        write_png(
            (
                validation_input[timestep],
                validation_target[timestep],
                predictions[timestep],
            ),
            "./output/prediction" + str(timestep) + ".png",
            pretty=True,
        )
