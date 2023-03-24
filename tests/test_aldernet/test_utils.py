"""Run unittests for all modules."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS Distributed under the
# terms of the BSD 3-Clause License. SPDX-License-Identifier: BSD-3-Clause Standard
# library
# Standard library
import os
import socket
import unittest
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

# Third-party
import keras  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
import xarray as xr
from pyprojroot import here  # type: ignore
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler

# First-party
# First-party Import the functions to test
from aldernet.utils import Batcher  # type: ignore
from aldernet.utils import cbr
from aldernet.utils import compile_generator
from aldernet.utils import create_optimizer
from aldernet.utils import define_filters
from aldernet.utils import down
from aldernet.utils import generate_report
from aldernet.utils import get_callbacks
from aldernet.utils import get_runtime_env
from aldernet.utils import get_scheduler
from aldernet.utils import get_tune_config
from aldernet.utils import load_data  # type: ignore
from aldernet.utils import rsync_mlruns
from aldernet.utils import setup_directories
from aldernet.utils import train_and_evaluate_model
from aldernet.utils import train_step
from aldernet.utils import train_with_ray_tune
from aldernet.utils import up
from aldernet.utils import validate_epoch
from aldernet.utils import write_png


class TestMyFunctions(unittest.TestCase):  # pylint: disable=R0902,R0904
    def setUp(self):
        # Set up any objects needed by the tests
        self.run_path = "test_run_path"
        self.settings = {
            "batch_size": 32,
            "members": 10,
            "device": {"cpu": 1},
            "noise_dim": 100,
            "add_weather": True,
            "shuffle": True,
            "input_species": "CORY",
            "tune_with_ray": True,
            "retrain_model": True,
            "target_species": "ALNU",
            "zoom": "",
            "epochs": 5,
        }
        self.tune_trial = "random_id"
        self.data_train = Batcher(
            xr.Dataset(
                {
                    "CORY": (["valid_time", "x", "y"], np.random.rand(8, 8, 8)),
                    "ALNU": (["valid_time", "x", "y"], np.random.rand(8, 8, 8)),
                }
            ),
            batch_size=32,
            add_weather=False,
            shuffle=True,
        )
        self.data_valid = Batcher(
            xr.Dataset(
                {
                    "CORY": (["valid_time", "x", "y"], np.random.rand(8, 8, 8)),
                    "ALNU": (["valid_time", "x", "y"], np.random.rand(8, 8, 8)),
                }
            ),
            batch_size=32,
            add_weather=False,
            shuffle=False,
        )
        self.sha = "test_sha"
        self.hostname_tsa = "tsa.example.com"
        self.hostname_nid = "nid.example.com"
        self.input_data = tf.constant(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32
        )
        self.target_data = tf.constant([[11], [12], [13], [14]], dtype=tf.float32)
        self.generator = tf.keras.Sequential()
        self.generator.add(tf.keras.layers.Dense(1, input_shape=(10,)))
        self.center = 40
        self.scale = 0.5

    def test_load_data(self):
        hostname = socket.gethostname()
        if "tsa" in hostname:
            # Test loading data from tsa.example.com
            data_train, data_valid = load_data(self.hostname_tsa, self.settings)
            self.assertIsInstance(data_train, xr.Dataset)
            self.assertIsInstance(data_valid, xr.Dataset)
        elif "nid" in hostname:
            # Test loading data from nid.example.com
            data_train, data_valid = load_data(self.hostname_nid, self.settings)
            self.assertIsInstance(data_train, xr.Dataset)
            self.assertIsInstance(data_valid, xr.Dataset)

        # Test raising ValueError for unknown hostname
        with self.assertRaises(ValueError):
            load_data("unknown.example.com", self.settings)

    @patch("aldernet.utils.train_with_ray_tune")
    @patch("aldernet.utils.train_without_ray_tune")
    @patch("aldernet.utils.load_pretrained_model")
    def test_train_and_evaluate_model(
        self,
        mock_load_pretrained_model,
        mock_train_without_ray_tune,
        mock_train_with_ray_tune,
    ):
        # Test retraining the model with Ray Tune
        train_and_evaluate_model(
            self.run_path, self.settings, self.data_train, self.data_valid, self.sha
        )
        mock_train_with_ray_tune.assert_called_once_with(
            self.run_path, self.settings, self.data_train, self.data_valid, self.sha
        )

        # Test retraining the model without Ray Tune
        self.settings["tune_with_ray"] = False
        train_and_evaluate_model(
            self.run_path, self.settings, self.data_train, self.data_valid, self.sha
        )
        mock_train_without_ray_tune.assert_called_once_with(
            self.settings, self.data_train, self.data_valid
        )

        # Test loading a pretrained model
        self.settings["retrain_model"] = False
        train_and_evaluate_model(
            self.run_path, self.settings, self.data_train, self.data_valid, self.sha
        )
        mock_load_pretrained_model.assert_called_once()

    def test_train_with_ray_tune(self):
        with patch("aldernet.utils.prepare_generator") as mock_prepare_generator, patch(
            "aldernet.utils.shutdown"
        ) as mock_shutdown, patch("aldernet.utils.init") as mock_init, patch(
            "aldernet.utils.tune.run"
        ) as mock_tune_run, patch(
            "aldernet.utils.rsync_mlruns"
        ):
            mock_prepare_generator.return_value = MagicMock()
            mock_tuner = MagicMock()
            mock_tuner.best_checkpoint.to_dict.return_value = {"model": MagicMock()}
            mock_tune_run.return_value = mock_tuner

            train_with_ray_tune(
                self.run_path, self.settings, self.data_train, self.data_valid, self.sha
            )

            mock_prepare_generator.assert_called_with(
                self.run_path, self.settings, self.data_train
            )
            mock_shutdown.assert_called()
            mock_init.assert_called_with(runtime_env=get_runtime_env())

    def test_get_runtime_env(self):
        # Act
        result = get_runtime_env()

        # Assert
        self.assertIsInstance(result, dict)
        self.assertIn("working_dir", result)
        self.assertIn("excludes", result)

    def test_get_scheduler(self):
        # Act
        result = get_scheduler(self.settings)

        # Assert
        self.assertIsInstance(result, ASHAScheduler)

    def test_get_tune_config(self):
        # Act
        result = get_tune_config()

        # Assert
        self.assertIsInstance(result, dict)
        self.assertIn("learning_rate", result)
        self.assertIn("beta_1", result)
        self.assertIn("beta_2", result)
        self.assertIn("mlflow", result)

    def test_get_callbacks(self):
        # Act
        result = get_callbacks(self.run_path, self.sha)

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MLflowLoggerCallback)

    @patch("subprocess.run")
    def test_rsync_mlruns(self, mock_run):
        # Act
        rsync_mlruns(self.run_path)

        # Assert
        mock_run.assert_called_once_with(
            f"rsync -avzh {self.run_path}/mlruns {str(here())}",
            shell=True,
            check=True,
        )

    @patch("aldernet.utils.train_model_simple")
    def test_train_without_ray_tune(self, mock_train_model_simple):
        # Arrange
        Mock()
        Mock()
        Mock()
        Mock()
        mock_train_model_simple.return_value = Mock()

    def test_generate_report(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Act
        generate_report(df, self.settings)

        # Assert
        with open(str(here()) + "/output/report.txt", "r", encoding="UTF-8") as f:
            lines = f.readlines()
            self.assertEqual(lines[0], "Training settings:\n")
            self.assertEqual(lines[1], "batch_size: 32\n")
            self.assertEqual(lines[2], "\n")
            self.assertEqual(lines[3], "Predictions and observed values:\n")
            self.assertEqual(lines[4], "   a  b\n")
            self.assertEqual(lines[5], "0  1  3\n")
            self.assertEqual(lines[6], "1  2  4members: 10\n")

    def test_define_filters(self):
        # Act
        result = define_filters("")

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 12)
        self.assertEqual(result[0], 4)

    def test_cbr(self):
        result = cbr(32)
        self.assertIsInstance(result, tf.keras.Sequential)

    def test_down(self):
        result = down(32)
        self.assertIsInstance(result, tf.keras.Sequential)

    def test_up(self):
        result = up(32)
        self.assertIsInstance(result, tf.keras.Sequential)

    def test_compile_generator(self):
        height = 256
        width = 256
        weather_features = 4
        noise_dim = 128
        filters = [64, 128, 256, 512, 512, 512, 512, 512]
        generator = compile_generator(
            height, width, weather_features, noise_dim, filters
        )
        self.assertIsInstance(generator, keras.Model)

    def test_write_png(self):
        # Create some example tensors
        input_tensor = tf.ones(shape=(4, 8, 8, 1)).numpy()
        target_tensor = tf.zeros(shape=(4, 8, 8, 1)).numpy()
        predicted_tensor = tf.random.uniform(shape=(4, 8, 8, 1)).numpy()
        image = (input_tensor, target_tensor, predicted_tensor)
        path = "test_image.png"
        pretty = True
        write_png(image, path, pretty)
        # Check if the file was created
        self.assertTrue(os.path.exists(path))

    def test_train_step(self):
        # Create some example inputs
        input_train = tf.ones(shape=(32, 8, 8, 1)).numpy()
        target_train = tf.zeros(shape=(32, 8, 8, 1)).numpy()
        weather_train = tf.random.uniform(shape=(32, 8, 8, 4)).numpy()
        noise_dim = 8
        add_weather = True
        filters = [4, 8, 8, 16, 16, 16, 16, 16]
        generator = compile_generator(8, 8, 4, 8, filters)
        optimizer_gen = tf.keras.optimizers.Adam(1e-4)
        loss = train_step(
            generator,
            optimizer_gen,
            input_train,
            target_train,
            weather_train,
            noise_dim,
            add_weather,
        )
        self.assertIsInstance(loss.numpy(), np.floating)

    def test_setup_directories(self):
        setup_directories(self.run_path, self.tune_trial)

        self.assertTrue(
            os.path.isdir(os.path.join(self.run_path, "viz", self.tune_trial))
        )
        self.assertTrue(
            os.path.isdir(os.path.join(self.run_path, "viz", "valid", self.tune_trial))
        )

    def test_create_optimizer(self):
        optimizer = create_optimizer(0.001, 0.9, 0.999)
        self.assertIsInstance(optimizer, tf.keras.optimizers.Optimizer)

    def test_validate_epoch(self):
        # create some mock data
        data_cory = {
            "CORY": (["valid_time", "x", "y"], np.random.rand(8, 8, 8)),
        }
        data_alnu = {
            "ALNU": (["valid_time", "x", "y"], np.random.rand(8, 8, 8)),
        }
        epoch = tf.Variable(1, dtype="int64")
        step = 1
        # create the combined xarray dataset
        data_xr_mock = xr.Dataset(data_vars={**data_cory, **data_alnu})

        data_valid = Batcher(data_xr_mock, add_weather=False, batch_size=32)

        input_layer = tf.keras.Input(shape=(None, 128, 128))
        same_dim_layer = tf.keras.layers.Lambda(lambda x: x)(input_layer)
        generator = tf.keras.models.Model(inputs=input_layer, outputs=same_dim_layer)

        loss_valid, step_valid = validate_epoch(
            generator, data_valid, 0, False, 0, 1, epoch, step, "./output/test", ""
        )
        self.assertIsInstance(loss_valid, np.ndarray)
        self.assertIsInstance(step_valid, int)


if __name__ == "__main__":
    unittest.main()
