"""Provide all required functions for the training script.

Functions:
~~~~~~~~~~
- build_unet(conv: bool=True) -> tf.keras.Model: builds a UNet model for predicting
    pollenconcentrations.

- cbr(filters, name=None) -> Sequential: Creates a convolutional block with batch
    normalization and leaky ReLU activation.

- compile_generator(height, width, weather_features, noise_dim, filters): Compiles a
    generator model using Tensorflow with the specified parameters.

- create_batcher(data: xarray.Dataset, batch_size: int, add_weather: bool, shuffle:
    bool) -> Batcher: creates a batcher object for the given data and parameters.

- create_optimizer(learning_rate: float, beta_1: float, beta_2: float)
    ->tf.keras.optimizers.Adam: creates an optimizer object for the model.

- define_filters(zoom: str) -> List[int]: Define the filters based on the provided zoom.

- down(filters, name=None) -> Sequential: Creates a downsampling block with
    convolutional layer, batch normalization, and leaky ReLU activation.

- generate_report(df: pd.DataFrame, settings: Dict[str, Union[str, int, float, bool]])
    -> None: Generate a report based on the provided dataframe and settings.

- get_callbacks(run_path: str, sha: str) -> List[ray.tune.callback.TrialCallback]: Get
    the callbacks based on the provided run path and sha.

- get_scheduler(settings: Dict[str, Union[str, int, float, bool]]) ->
    ray.tune.schedulers.asha.ASHAScheduler: Get the ASHAScheduler based on the provided
    settings.

- get_tune_config(run_path: str) -> Dict[str, Union[str, Dict[str, str], float]]: Get
    the Tune configuration based on the provided run path.

- load_data(settings: Dict[str, Union[str, int, float, bool]]) ->
    Tuple[xr.Dataset, xr.Dataset]: Load data based on the provided settings.

- load_pretrained_model() -> keras.engine.sequential.Sequential: Load a pre-trained
    model.

- predict_season(best_model: tf.keras.Model, data_valid: xr.Dataset, noise_dim: int,
    add_weather: bool) -> np.ndarray: predicts the pollen season with the best model and
    returns a numpy array of predicted values.

- prepare_generator(run_path: str, settings: Dict[str, Union[str, int, float, bool]],
    data_train: xr.Dataset) -> keras.engine.training.Model: Prepare the generator based
    on the provided settings and data.

- read_scaling_data() -> Tuple[float, float]: reads scaling data from a file and returns
    a tuple containing the center and scale.

- rsync_mlruns(run_path: str) -> None: Rsync the mlruns based on the provided run path.

- save_generator_summary_and_plot(run_path: str, generator: keras.engine.training.Model)
    -> None: Save the generator summary and plot based on the provided run path and
    generator.

- save_predictions_and_generate_report(settings: Dict[str, Union[str, int, float,
    bool]], best_model: keras.engine.sequential.Sequential, data_valid: xr.Dataset) ->
    None: Save the predictions and generate the report based on the provided settings,
    best_model, and data_valid.

- setup_directories(run_path: str, tune_trial: str) -> None: creates directories for
    visualizations.

- setup_output_directory(settings: Dict[str, Union[str, int, float, bool]]) -> str: Set
    up the output directory based on the provided settings.


- train_and_evaluate_model(run_path: str, settings: Dict[str, Union[str, int, float,
    bool]], data_train: xr.Dataset, data_valid: xr.Dataset, sha: str) ->
    keras.engine.sequential.Sequential: Train and evaluate the model based on the
    provided settings, data, and run path.

- train_epoch(generator: tf.keras.Model, optimizer_gen: tf.keras.optimizers.Optimizer,
    data_train:Batcher, noise_dim: int, add_weather: bool, center: float, scale: float,
    epoch: tf.Variable,step: tf.Variable, run_path: str, tune_trial: str) -> np.ndarray:
    trains the model for one epoch and returns a numpy array of loss values.

- train_model_simple(data_train: xr.Dataset, data_valid: xr.Dataset, epochs: int,
    add_weather: bool, conv: bool=True) -> tf.keras.Model: trains a simple model for
    predicting pollen concentrations and returns the trained model.

- train_model(config: Dict[str, float], generator: tf.keras.Model, data_train:
    Batcher,data_valid: Batcher, run_path: str, noise_dim: int, add_weather: bool,
    shuffle: bool) -> None: trains the model using a configuration dictionary and saves
    the best model.

- train_step(generator, optimizer_gen, input_train, target_train, weather_train,
    noise_dim, add_weather): Performs one training step for the generator of a NNet to
    generate images of pollen concentrations in the air given an input image of trees
    and weather data.

- train_with_ray_tune(run_path: str, settings: Dict[str, Union[str, int, float, bool]],
    data_train: xr.Dataset, data_valid: xr.Dataset, sha: str) ->
    keras.engine.sequential.Sequential: Train the model using Ray Tune based on the
    provided settings, data, and run path.

- train_without_ray_tune(settings: Dict[str, Union[str, int, float, bool]], data_train:
    xr.Dataset, data_valid: xr.Dataset) -> keras.engine.sequential.Sequential: Train the
    model without Ray Tune based on the provided settings and data.

- up(filters, name=None) -> Sequential: Creates an upsampling blockwith transposed
    convolutional layer, batch normalization, and leaky ReLU activation.

- validate_epoch(generator: tf.keras.Model, data_valid: Batcher, noise_dim: int,
    add_weather: bool, center: float, scale: float, epoch: tf.Variable, step_valid: int,
    run_path: str, tune_trial: str) -> Tuple[np.ndarray, int]: validates the model for
    one epoch and returns a tuple containing a numpy array of loss values and the step
    number.


Attributes:
~~~~~~~~~~~

All functions and classes defined in this script.

COPYRIGHT (c) 2022 MeteoSwiss, contributors listed in AUTHORS. Distributed under the
terms of the BSD 3-Clause License. SPDX-License-Identifier: BSD-3-Clause

This docstring was autogenerated by GPT-4.

"""

# pylint: disable=C0302
# pyright:reportOptionalMemberAccess=false,reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false

# Standard library
import datetime
import math
import os
import subprocess
import time
from contextlib import redirect_stdout
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
import xarray as xr
from genericpath import exists
from pyprojroot import here
from ray import init
from ray import shutdown
from ray import train
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model  # type: ignore

# First-party
from aldernet.data.data_utils import Batcher
from aldernet.data.data_utils import Stations


def load_data(settings, debug_mode=False, debug_data_size=100):
    """Load training and validation data.

    Args:
        settings (dict): Dictionary of settings. debug_mode (bool): Whether to load a
        reduced dataset for debugging. debug_data_size (int): Number of samples to load
        in debug mode.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: A tuple containing the training and validation
        data.

    """
    if exists(settings["data_path"]):
        data_train = xr.open_zarr(
            settings["data_path"] + "/data_train.zarr",
            consolidated=True,
        )
        data_valid = xr.open_zarr(
            settings["data_path"] + "/data_valid.zarr",
            consolidated=True,
        )
        if debug_mode:
            data_train = data_train.isel(valid_time=slice(0, debug_data_size))
            data_valid = data_valid.isel(valid_time=slice(0, debug_data_size))
    else:
        raise ValueError("Unknown data path. Data cannot be loaded.")

    return data_train, data_valid


def setup_output_directory(settings):
    """Create a new output directory and return the path to the new directory.

    Args:
        settings (dict): Dictionary of settings.

    Returns:
        str: The path to the new output directory.

    """
    run_path = (
        str(here()) + "/output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if settings["retrain_model"]:
        Path(run_path + "/viz/valid").mkdir(parents=True, exist_ok=True)
    return run_path


def train_and_evaluate_model(run_path, settings, sha, debug_mode=False):
    """Train and evaluate a neural network model using the specified data and settings.

    Args:
        run_path (str): Path to the output directory for storing results. settings
        (dict): Dictionary of settings. sha (str): A string containing the Git commit
        hash for the current code. debug_mode (bool): Whether to run in debug mode with
        reduced dataset and more logs.

    Returns:
        The trained neural network model.

    """
    data_train, data_valid = load_data(settings, debug_mode=debug_mode)
    # Additional debug-specific logic here if needed
    if settings["retrain_model"]:
        best_model = train_with_ray_tune(
            run_path, settings, data_train, data_valid, sha
        )
    else:
        best_model = load_pretrained_model(settings)
    return best_model


def train_with_ray_tune(run_path, settings, data_train, data_valid, sha):
    generator = prepare_generator(run_path, settings, data_train)

    shutdown()
    init(runtime_env=get_runtime_env())

    tuner = tune.run(
        tune.with_parameters(
            train_model,
            generator=generator,
            data_train=data_train,
            data_valid=data_valid,
            run_path=run_path,
            noise_dim=settings["noise_dim"],
            add_weather=settings["add_weather"],
            shuffle=settings["shuffle"],
        ),
        metric="loss",
        mode="min",
        num_samples=settings["members"],
        scheduler=get_scheduler(settings),
        resources_per_trial=settings["device"],
        config=get_tune_config(),
        local_dir=run_path,
        keep_checkpoints_num=1,
        callbacks=get_callbacks(run_path, sha),
    )

    best_trial = tuner.get_best_trial("loss", "min", "all")
    print(best_trial)
    best_checkpoint = tuner.get_best_checkpoint(best_trial, "loss", "min")
    print(best_checkpoint)

    best_model = load_model(best_checkpoint.to_directory())

    rsync_mlruns(run_path)
    return best_model


def prepare_generator(run_path, settings, data_train):
    """Prepare the generator model for training the NNet.

    Args:
        run_path (str): Path to the output directory for storing results. settings
        (dict): Dictionary of settings. data_train (xarray.Dataset): Training data.

    Returns:
        tensorflow.python.keras.engine.functional.Functional: The compiled generator
        model.

    """
    height = data_train[settings["input_species"]].shape[1]
    width = data_train[settings["input_species"]].shape[2]
    if settings["add_weather"]:
        weather_features = len(
            data_train.drop_vars(
                (settings["target_species"], settings["input_species"])
            ).data_vars
        )
    else:
        weather_features = 0
        filters = define_filters(settings["zoom"])
        generator = compile_generator(
            height, width, weather_features, settings["noise_dim"], filters
        )
    save_generator_summary_and_plot(run_path, generator)
    return generator


def save_generator_summary_and_plot(run_path, generator):
    """Save a summary of the generator model and a visualization of its architecture.

    Args:
        run_path (str): Path to the output directory for storing results.
        generator(tensorflow.python.keras.engine.functional.Functional): The generator
        model to summarize and plot.

    """
    with open(run_path + "/generator_summary.txt", "w", encoding="UTF-8") as handle:
        with redirect_stdout(handle):
            generator.summary()
            plot_model(
                generator, to_file=run_path + "/generator.png", show_shapes=True, dpi=96
            )


def get_runtime_env():
    """Get the runtime environment for Ray Tune.

    Returns:
        dict: A dictionary containing the working directory and excluded files and
        directories.

    """
    return {
        "working_dir": str(here()),
        "excludes": ["data/", ".git/", "images/", "core"],
    }


def get_scheduler(settings):
    """Get the scheduler for Ray Tune.

    Args:
        settings (dict): Dictionary of settings.

    Returns:
        ray.tune.schedulers.AsyncHyperBandScheduler: The scheduler for hyperparameter
        tuning.

    """
    return ASHAScheduler(
        time_attr="training_iteration",
        max_t=settings["epochs"],
        grace_period=1,
        reduction_factor=3,
    )


def get_tune_config():
    """Get the configuration settings for Ray Tune.

    Args:
        None

    Returns:
        dict: A dictionary containing the learning rate, beta values, and MLflow
        settings.

    """
    return {
        "learning_rate": tune.choice([0.001]),
        "beta_1": tune.choice([0.9, 0.95]),
        "beta_2": tune.choice([0.98, 0.99]),
        "mlflow": {
            "experiment_name": "Aldernet",
            "tracking_uri": mlflow.get_tracking_uri(),
        },
    }


def get_callbacks(run_path, sha):
    """Get the MLflow logger callback for logging the experiment results.

    Args:
        run_path (str): Path to the output directory for storing results. sha (str): A
        string containing the Git commit hash for the current code.

    Returns:
        list: A list containing the MLflow logger callback.

    """
    return [
        MLflowLoggerCallback(
            experiment_name="Aldernet",
            tags={"git_commit": sha},
            tracking_uri=run_path + "/mlruns",
            save_artifact=True,
        )
    ]


def rsync_mlruns(run_path):
    """Copy the MLflow run data to the current working directory.

    Args:
        run_path (str): Path to the output directory for storing results.

    """
    rsync_cmd = "rsync" + " -avzh " + run_path + "/mlruns" + " " + str(here())
    subprocess.run(rsync_cmd, shell=True, check=True)


def load_pretrained_model(settings):
    """Load a pretrained neural network model.

    Returns:
        tensorflow.python.keras.engine.functional.Functional: The pretrained neural
        network model.

    """
    return Checkpoint.from_directory(settings["checkpoint"]).to_dict()["model"]


def save_predictions_and_generate_report(settings, best_model, data_valid):
    """Save the predicted and observed values to a CSV file and generate a report.

    Args:
        settings (dict): Dictionary of settings.
        best_model(tensorflow.python.keras.engine.functional.Functional): The trained
        neural network model.
        data_valid (xarray.Dataset): Validation data.

    """
    predictions = predict_season(
        best_model, data_valid, settings["noise_dim"], settings["add_weather"]
    )
    with open(str(here()) + "/data/scaling.txt", "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
        center = float(lines[0].split(" ")[-1])
        scale = float(lines[1].split(" ")[-1])

    data_valid[settings["target_species"]] = (data_valid.dims, np.squeeze(predictions))
    data_valid_out = np.maximum(
        0, data_valid[settings["target_species"]] * scale + center
    )
    data_valid_out.to_netcdf(str(here()) + "/data/pollen_ml.nc")  # type: ignore

    if settings["target_species"] == "ALNU":
        target_species_name = "Alnus"
    else:
        target_species_name = ""

    station_values = data_valid_out.values[  # type:ignore
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


def generate_report(df, settings):
    """Generate a report summarizing the experiment results.

    Args:
        df (pandas.core.frame.DataFrame): A dataframe containing the predicted and
        observed values.
        settings (dict): Dictionary of settings.

    """
    report_path = str(here()) + "/output/report.txt"
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("Training settings:\n")
        for key, value in settings.items():
            report.write(f"{key}: {value}\n")
            report.write("\nPredictions and observed values:\n")
            report.write(df.to_string())


def define_filters(zoom):
    """Define the filters for the generator model based on the specified zoom level.

    Args:
        zoom (str): The zoom level.

    Returns:
        list: A list containing the filter values for the generator model.

    """
    filters = [64, 128, 256, 512, 1024, 1024, 512, 768, 640, 448, 288, 352]

    if zoom == "":
        filters[:] = [int(x / 16) for x in filters]

    return filters


def tf_setup():
    """Set up Tensorflow to use GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def cbr(filters, name=None) -> Sequential:
    """Construct a Convolution-BatchNorm-ReLU block using the Keras Sequential API.

    Args:
        filters (int): Number of filters in the Conv2D layer. name (str): Optional name
        for the block.

    Returns:
        Sequential: A Keras Sequential model representing the CBR block.

    """
    block = tf.keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def down(filters, name=None) -> Sequential:
    """Construct a down-sampling block using the Keras Sequential API.

    Args:
        filters (int): Number of filters in the Conv2D layer. name (str): Optional name
        for the block.

    Returns:
        Sequential: A Keras Sequential model representing the down-sampling block.

    """
    block = tf.keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    # block.add(layers.MaxPooling2D((2, 2), padding="same"))
    # block.add(layers.Dropout(0.05))

    return block


def up(filters, name=None) -> Sequential:
    """Construct an up-sampling block using the Keras Sequential API.

    Args:
        filters (int): Number of filters in the Conv2DTranspose layer. name (str):
        Optional name for the block.

    Returns:
        Sequential: A Keras Sequential model representing the up-sampling block.

    """
    block = tf.keras.Sequential(name=name)
    block.add(
        layers.Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )
    )
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def compile_generator(height, width, weather_features, noise_dim, filters):
    """Compile a generator model for image-to-image translation using Keras API.

    Args:
        height (int): Height of the input images. width (int): Width of the input
        images. weather_features (int): Number of weather features in the weather input.
        noise_dim (int): Dimensionality of the noise input. filters (List[int]): Number
        of filters in each layer of the generator.

    Returns:
        Model: A Keras Model representing the compiled generator model.

    """
    image_input = tf.keras.Input(shape=[height, width, 1], name="image_input")
    noise_input = weather_input = tf.keras.Input(shape=[])
    if weather_features > 0:
        weather_input = tf.keras.Input(
            shape=[height, width, weather_features], name="weather_input"
        )
        inputs = layers.Concatenate(name="inputs-concat")([image_input, weather_input])
    else:
        inputs = image_input
    block = cbr(filters[0], "pre-cbr-1")(inputs)

    u_skip_layers = [block]
    for ll in range(1, len(filters) // 2):
        block = down(filters[ll], f"down_{ll}-down")(block)
        # Collect U-Net skip connections
        u_skip_layers.append(block)
    height = block.shape[1]
    width = block.shape[2]
    if noise_dim > 0:
        noise_channels = 128
        noise_input = tf.keras.Input(shape=noise_dim, name="noise_input")
        noise = layers.Dense(
            height * width * noise_channels,
        )(noise_input)
        noise = layers.Reshape((height, width, -1))(noise)
        block = layers.Concatenate(name="add-noise")([block, noise])
    u_skip_layers.pop()

    for ll in range(len(filters) // 2, len(filters) - 1):
        block = up(filters[ll], f"up_{(len(filters) - ll - 1)}-up")(block)
        if block.shape[slice(1, 2)] != u_skip_layers[-1].shape[slice(1, 2)]:
            block = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(
                block
            )
        # Connect U-Net skip
        block = layers.Concatenate(name=f"up_{(len(filters) - ll - 1)}-concatenate")(
            [block, u_skip_layers.pop()]
        )

    block = cbr(filters[-1], "post-cbr-1")(block)

    pollen = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="tanh",
        name="output",
    )(block)
    if weather_features > 0 and noise_dim > 0:
        return tf.keras.Model(
            inputs=[noise_input, image_input, weather_input], outputs=pollen
        )
    elif noise_dim <= 0 < weather_features:
        return tf.keras.Model(inputs=[image_input, weather_input], outputs=pollen)
    elif weather_features <= 0 < noise_dim:
        return tf.keras.Model(inputs=[noise_input, image_input], outputs=pollen)
    else:
        return tf.keras.Model(inputs=[image_input], outputs=pollen)


def write_png(image, path, pretty):
    """Write an image in PNG format to a file.

    Args:
        image (Tensor): A 3-tuple of Tensors representing the input, target, and
        predicted images. path (str): Path to the output file. pretty (bool): Whether to
        create a pretty visualization of the image.

    Returns:
        None

    """
    if pretty:
        minmin = min(image[0].min(), image[1].min(), image[2].min())
        maxmax = min(max(image[0].max(), image[1].max(), image[2].max()), 500)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.1), dpi=150)
        ax1.imshow(
            image[0][:, :, 0], cmap="viridis", vmin=minmin, vmax=maxmax, aspect="auto"
        )
        ax2.imshow(
            image[1][:, :, 0], cmap="viridis", vmin=minmin, vmax=maxmax, aspect="auto"
        )
        im3 = ax3.imshow(
            image[2][:, :, 0], cmap="viridis", vmin=minmin, vmax=maxmax, aspect="auto"
        )
        for ax in (ax1, ax2, ax3):
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        ax1.axes.set_title("Input")
        ax2.axes.set_title("Target")
        ax3.axes.set_title("Prediction")
        fig.subplots_adjust(right=0.85, top=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im3, cax=cbar_ax)
        plt.savefig(path)
        plt.close(fig)
    else:
        image = tf.concat(image, 1) * 10
        image = tf.cast(image, tf.uint8)
        png = tf.image.encode_png(image)
        tf.io.write_file(path, png)


def train_step(  # pylint: disable=R0913
    generator,
    optimizer_gen,
    input_train,
    target_train,
    weather_train,
    noise_dim,
    add_weather,
):
    """Perform one training step of the generator of a NNet using the given inputs.

    Args:
        generator (Model): The Keras Model representing the generator. optimizer_gen
        (Optimizer): The optimizer for the generator. input_train (Tensor): The input
        tensor for the generator. target_train (Tensor): The target tensor for the
        generator. weather_train (Tensor): The weather tensor for the generator.
        noise_dim (int): Dimensionality of the noise input. add_weather (bool): Whether
        to add the weather input to the generator.

    Returns:
        float: The loss of the generator on the given inputs.

    """
    noise = tf.random.normal(shape=[])
    if noise_dim > 0:
        noise = tf.random.normal([input_train.shape[0], noise_dim])
    with tf.GradientTape() as tape_gen:
        if add_weather and noise_dim > 0:
            generated = generator([noise, input_train, weather_train])
        elif add_weather and noise_dim <= 0:
            generated = generator([input_train, weather_train])
        elif not add_weather and noise_dim > 0:
            generated = generator([noise, input_train])
        else:
            generated = generator([input_train])
        loss = tf.math.reduce_mean(tf.math.abs(generated - target_train))
        # loss = tf.math.reduce_mean(tf.math.squared_difference(generated, alder))
        gradients_gen = tape_gen.gradient(loss, generator.trainable_variables)

    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return loss


def create_batcher(data, batch_size, add_weather, shuffle):
    """Create a Batcher object for the given data.

    Args:
        data (Tuple[Tensor]): A tuple of Tensors representing the input and target data.
        batch_size (int): Batch size for training. add_weather (bool): Whether to add
        weather data to the input. shuffle (bool): Whether to shuffle the data during
        training.

    Returns:
        Batcher: A Batcher object for the given data.

    """
    return Batcher(
        data, batch_size=batch_size, add_weather=add_weather, shuffle=shuffle
    )


def setup_directories(run_path, tune_trial):
    """Set up the directory structure for saving visualization images.

    Args:
        run_path (str): Path to the run directory. tune_trial (str): Trial ID for
        hyperparameter tuning.

    Returns:
        None

    """
    Path(run_path + "/viz/" + tune_trial).mkdir(parents=True, exist_ok=True)
    Path(run_path + "/viz/valid/" + tune_trial).mkdir(parents=True, exist_ok=True)


def read_scaling_data():
    """Read and returns the center and scale values from scaling.txt file.

    Returns:
        Tuple[float, float]: A tuple of center and scale values.

    """
    with open(str(here()) + "/data/scaling.txt", "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
        center = float(lines[0].split(": ")[1])
        scale = float(lines[1].split(": ")[1])
    return center, scale


def create_optimizer(learning_rate, beta_1, beta_2):
    """Create and returns an Adam optimizer with given hyperparameters.

    Args:
        learning_rate (float): The learning rate for the optimizer. beta_1 (float): The
        exponential decay rate for the first moment estimates. beta_2 (float): The
        exponential decay rate for the second-moment estimates.

    Returns:
        tf.keras.optimizers.Optimizer: An instance of the Adam optimizer.

    """
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=1e-08,
    )


def train_epoch(
    generator,
    optimizer_gen,
    data_train,
    noise_dim,
    add_weather,
    center,
    scale,
    epoch,
    step,
    run_path,
    tune_trial,
):  # pylint: disable=R0913,R0914
    """Trains the generator for one epoch and returns the loss values.

    Args:
        generator (tf.keras.Model): The generator model. optimizer_gen
        (tf.keras.optimizers.Optimizer): The generator optimizer. data_train (Batcher):
        The training data batcher. noise_dim (int): The dimensionality of the noise
        vector. add_weather (bool): If True, the weather data is included in training.
        center (float): The center value for the scaling. scale (float): The scale value
        for the scaling. epoch (tf.Variable): The current epoch. step (tf.Variable): The
        current step. run_path (str): The path to the directory where the output files
        will be stored. tune_trial (str): The name of the current Tune trial.

    Returns:
        np.ndarray: An array of loss values.

    """
    start = time.time()
    loss_report = np.zeros(0)
    for i in range(math.floor(data_train.x.shape[0] / data_train.batch_size)):
        if not add_weather:
            hazel_train, alder_train = data_train[i]
        else:
            hazel_train, weather_train, alder_train = data_train[i]

        print(epoch.numpy(), "-", step.numpy(), flush=True)

        if add_weather:
            loss_report = np.append(
                loss_report,
                train_step(
                    generator,
                    optimizer_gen,
                    hazel_train,
                    alder_train,
                    weather_train if add_weather else None,
                    noise_dim,
                    add_weather,
                ).numpy(),
            )
        else:
            loss_report = np.append(
                loss_report,
                train_step(
                    generator,
                    optimizer_gen,
                    hazel_train,
                    alder_train,
                    None,
                    noise_dim,
                    add_weather,
                ).numpy(),
            )
        if noise_dim > 0:
            noise_train = tf.random.normal([hazel_train.shape[0], noise_dim])
            generated_train = generator(
                [noise_train, hazel_train] + ([weather_train] if add_weather else [])
            )
        else:
            generated_train = generator(
                [hazel_train] + ([weather_train] if add_weather else [])
            )

        index = np.random.randint(hazel_train.shape[0])

        viz = (
            hazel_train[index] * scale + center,
            alder_train[index] * scale + center,
            generated_train[index].numpy() * scale + center,
        )

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
        f"Time taken for epoch {epoch.numpy()} is {time.time() - start} sec\n",
        flush=True,
    )
    return loss_report


def validate_epoch(
    generator,
    data_valid,
    noise_dim,
    add_weather,
    center,
    scale,
    epoch,
    step_valid,
    run_path,
    tune_trial,
):  # pylint: disable=R0913,R0914
    """Evaluate the generator for one epoch on validation data and return the loss.

    Args:
        generator (tf.keras.Model): The generator model. data_valid (Batcher): The
        validation data batcher. noise_dim (int): The dimensionality of the noise
        vector. add_weather (bool): If True, the weather data is included in training.
        center (float): The center value for the scaling. scale (float): The scale value
        for the scaling. epoch (tf.Variable): The current epoch. step_valid (int): The
        current validation step. run_path (str): The path to the directory where the
        output files will be stored. tune_trial (str): The name of the current Tune
        trial.

    Returns:
        Tuple[np.ndarray, int]: A tuple of loss values and the current validation step.

    """
    loss_valid = np.zeros(0)
    for i in range(math.floor(data_valid.x.shape[0] / data_valid.batch_size)):
        if not add_weather:
            hazel_valid, alder_valid = data_valid[i]
        else:
            hazel_valid, weather_valid, alder_valid = data_valid[i]

        if noise_dim > 0:
            noise_valid = tf.random.normal([hazel_valid.shape[0], noise_dim])
            generated_valid = generator(
                [noise_valid, hazel_valid] + ([weather_valid] if add_weather else [])
            )
        else:
            generated_valid = generator(
                [hazel_valid] + ([weather_valid] if add_weather else [])
            )

        index = np.random.randint(hazel_valid.shape[0])

        viz = (
            hazel_valid[index] * scale + center,
            alder_valid[index] * scale + center,
            generated_valid[index].numpy() * scale + center,
        )

        write_png(
            viz,
            run_path
            + "/viz/valid/"
            + tune_trial
            + str(epoch.numpy())
            + "-"
            + str(step_valid)
            + ".png",
            pretty=True,
        )
        step_valid += 1
        loss_valid = np.append(
            loss_valid,
            tf.math.reduce_mean(tf.math.abs(generated_valid - alder_valid)).numpy(),
        )

    return loss_valid, step_valid


def train_model(
    config, generator, data_train, data_valid, run_path, noise_dim, add_weather, shuffle
):  # pylint: disable=R0913,R0914
    # Setup for training
    data_train = create_batcher(data_train, 32, add_weather, shuffle)
    data_valid = create_batcher(data_valid, 32, add_weather, shuffle)
    mlflow.set_tracking_uri(run_path + "/mlruns")
    mlflow.set_experiment("Aldernet")
    tune_trial = train.get_context().get_trial_name() + "/"
    setup_directories(run_path, tune_trial)
    epoch = tf.Variable(1, dtype="int64")
    step = tf.Variable(1, dtype="int64")
    step_valid = 1
    center, scale = read_scaling_data()
    optimizer_gen = create_optimizer(
        config["learning_rate"], config["beta_1"], config["beta_2"]
    )

    while True:
        # Training and validation logic
        loss_report = train_epoch(
            generator,
            optimizer_gen,
            data_train,
            noise_dim,
            add_weather,
            center,
            scale,
            epoch,
            step,
            run_path,
            tune_trial,
        )
        loss_valid, step_valid = validate_epoch(
            generator,
            data_valid,
            noise_dim,
            add_weather,
            center,
            scale,
            epoch,
            step_valid,
            run_path,
            tune_trial,
        )

        # Save model checkpoint
        checkpoint_dir = os.path.join(
            run_path, tune_trial, "checkpoints", f"epoch_{epoch.numpy()}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save your model's state to checkpoint_path
        generator.save(checkpoint_dir)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

        # Report metrics and checkpoint path to Ray Tune
        train.report(
            {
                "iterations": epoch,
                "loss_valid": loss_valid.mean(),
                "loss": loss_report.mean(),
            },
            checkpoint=checkpoint,
        )


def build_unet(conv=True):  # pylint: disable=R0914
    """Build a U-Net model.

    Args:
        conv (bool): Whether to use convolutional blocks.

    Returns:
        tf.keras.Model: U-Net model.

    """
    inputs = tf.keras.Input(shape=(None, None, None))

    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    if conv:
        conv_block_specs = [
            (16, 0.1),
            (16, 0.1),
            (32, 0.1),
            (32, 0.2),
            (64, 0.2),
            (64, 0.2),
            (128, 0.2),
            (128, 0.3),
        ]
        conv_blocks = []
        pool_layers = []

        for i, (filters, dropout_rate) in enumerate(conv_block_specs):
            block = tf.keras.Sequential(name=f"conv_block_{i}")
            block.add(
                tf.keras.layers.Conv2D(
                    filters,
                    (3, 3),
                    activation=tf.keras.activations.elu,
                    kernel_initializer="he_normal",
                    padding="same",
                )
            )
            block.add(tf.keras.layers.Dropout(dropout_rate))
            block.add(
                tf.keras.layers.Conv2D(
                    filters,
                    (3, 3),
                    activation=tf.keras.activations.elu,
                    kernel_initializer="he_normal",
                    padding="same",
                )
            )
            conv_blocks.append(block)

            if i < len(conv_block_specs) - 1:
                pool_layers.append(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

        c = s
        for conv_block, pool_layer in zip(conv_blocks, pool_layers):
            c = conv_block(c)
            p = pool_layer(c)

        c = conv_blocks[-1](p)

        for i, (filters, dropout_rate) in enumerate(reversed(conv_block_specs)):
            if i == len(conv_block_specs) - 1:
                break
            u = tf.keras.layers.Conv2DTranspose(
                filters // 2, (2, 2), strides=(2, 2), padding="same"
            )(c)
            if u.shape != conv_blocks[-i - 2].output.shape:
                u = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(u)
            u = tf.keras.layers.concatenate([u, conv_blocks[-i - 2].output])
            c = conv_blocks[-i - 2](u)
            c = tf.keras.layers.Dropout(dropout_rate)(c)
            c = tf.keras.layers.Conv2D(
                filters // 2,
                (3, 3),
                activation=tf.keras.activations.elu,
                kernel_initializer="he_normal",
                padding="same",
            )(c)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    else:
        model = tf.keras.models.Sequential()
        model.add(layers.Dense(1, input_shape=([None, None, None])))
        model.add(
            layers.Conv2D(
                filters=2,
                kernel_size=4,
                strides=1,
                padding="same",
                use_bias=False,
            )
        )
        model.add(layers.Dense(1, activation="linear"))

    model.summary()

    return model


def train_model_simple(data_train, data_valid, epochs, add_weather, conv=True):
    """Train a simple U-Net model on `data_train` and validate on `data_valid`.

    Args:
        data_train (Batcher): Training data. data_valid (Batcher): Validation data.
        epochs (int): Number of epochs to train the model for. add_weather (bool):
        Whether or not to include weather data. conv (bool, optional): Whether to use
        convolutional layers (default is True).

    Returns:
        model (keras.Model): The trained U-Net model.

    """
    with open(str(here()) + "/data/scaling.txt", "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
        center = float(lines[0].split(": ")[1])
        scale = float(lines[1].split(": ")[1])

    if add_weather:
        data_train.x = xr.concat([data_train.x, data_train.weather], dim="var")
        data_valid.x = xr.concat([data_valid.x, data_valid.weather], dim="var")

    model = build_unet(conv=conv)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(data_train, epochs=epochs)

    predictions = model.predict(data_valid)

    for timestep in range(0, predictions.shape[0], 100):
        inputs = (
            data_valid.x[timestep].values * scale + center,
            data_valid.y[timestep].values * scale + center,
            predictions[timestep] * scale + center,
        )
        path = str(here()) + f"/output/prediction{timestep}.png"
        write_png(inputs, path=path, pretty=True)

    return model


def predict_season(best_model, data_valid, noise_dim, add_weather):
    """Use the trained `best_model` to predict the full pollen season for `data_valid`.

    Args:
        best_model (keras.Model): The trained U-Net model. data_valid (xarray.Dataset):
        The validation data. noise_dim (int): The dimensionality of the noise vector.
        add_weather (bool): Whether or not to include weather data.

    Returns:
        predictions (ndarray): The predicted pollen season.

    """
    data_valid = Batcher(
        data_valid.sortby("valid_time"),
        batch_size=32,
        add_weather=add_weather,
        shuffle=False,
    )

    if noise_dim > 0:
        noise_season = tf.random.normal([data_valid.x.shape[0], noise_dim])
        if add_weather:
            predictions = best_model.predict(
                [noise_season, data_valid.x, data_valid.weather]
            )
        else:
            predictions = best_model.predict([noise_season, data_valid.x])
    else:
        if add_weather:
            predictions = best_model.predict([data_valid.x, data_valid.weather])
        else:
            predictions = best_model.predict(data_valid.x)
    return predictions
