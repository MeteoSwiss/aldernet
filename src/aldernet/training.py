"""Train a U-Net based on COSMO-1e input data.

This script loads and processes input data and trains a U-Net model to predict target
species data using COSMO-1e input data. The model is trained and evaluated using the
training and validation data, and the best model is saved. Predictions are then made
using the saved model on the validation data, and a report is generated with the
predictions. The report is then converted to an html file using an R script.

Modules:
~~~~~~~~
    - subprocess: Provides a way to run the R script to convert the report to an html
        file.
    - git: Provides a way to get the SHA of the current git commit.
    - pyprojroot: Provides a way to get the root directory of the project.
    - tensorflow: Provides a way to set up the tensorflow backend and random seed.

First-party Modules:
~~~~~~~~~~~~~~~~~~~~
    - aldernet.utils.load_data: Provides a way to load the training and validation data.
    - aldernet.utils.save_predictions_and_generate_report: Provides a way to save the
        predictions and generate a report with them.
    - aldernet.utils.setup_output_directory: Provides a way to set up the output
        directory.
    - aldernet.utils.tf_setup: Provides a way to set up the tensorflow backend.
    - aldernet.utils.train_and_evaluate_model: Provides a way to train and evaluate the
      model.

Variables:
~~~~~~~~~~
    - repo: The git repository object for the project.
    - sha: The SHA of the current git commit.
    - settings: A dictionary containing the settings for the script. The keys are:
        - "input_species": The species to use for input data.
        - "target_species": The species to predict.
        - "retrain_model": Whether to retrain the model or use a saved model.
        - "zoom": The zoom level to use for the input data.
        - "noise_dim": The dimension of the noise vector to use for the generator.
        - "epochs": The number of epochs to train the model for.
        - "shuffle": Whether to shuffle the training data before each epoch.
        - "add_weather": Whether to add weather data to the input data.
        - "conv": Whether to use convolutional layers instead of dense layers.
        - "members": The number of ensemble members to use.
        - "device": A dictionary containing the device to use for training. The keys
            are:
            - "gpu": The number of the GPU to use for training.

    - data_train: The training data.
    - data_valid: The validation data.
    - run_path: The path to the output directory for the script.
    - best_model: The best model obtained during training.
"""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# This optimized code has been refactored, linted, formatted, modularized, and cleaned
# up by GPT-4 to improve readability and maintainability.


# Standard library
import subprocess

# Third-party
import git
import yaml
from pyprojroot import here  # type: ignore
from tensorflow import random  # type: ignore

# First-party
from aldernet.utils import load_data
from aldernet.utils import save_predictions_and_generate_report
from aldernet.utils import setup_output_directory
from aldernet.utils import tf_setup
from aldernet.utils import train_and_evaluate_model


def main():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    debug_mode = True

    with open(
        str(here()) + "/src/aldernet/hyperparameters.yaml", "r", encoding="utf-8"
    ) as f:
        settings = yaml.safe_load(f)

    tf_setup()
    random.set_seed(1)

    _, data_valid = load_data(settings)
    run_path = setup_output_directory(settings)
    best_model = train_and_evaluate_model(run_path, settings, sha, debug_mode)
    save_predictions_and_generate_report(settings, best_model, data_valid)

    with subprocess.Popen(
        ["Rscript", "--vanilla", str(here()) + "/notebooks/rmd2html.R"], shell=False
    ):
        print("Creating html verification report.")


if __name__ == "__main__":
    main()
