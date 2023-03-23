"""Run unittests for all modules."""

# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# Standard library
import socket
import unittest
from unittest.mock import patch

# Third-party
import xarray as xr

# First-party
# Import the functions to test
from aldernet.utils import load_data  # type: ignore
from aldernet.utils import train_and_evaluate_model


class TestMyFunctions(unittest.TestCase):
    def setUp(self):
        # Set up any objects needed by the tests
        self.settings = {
            "zoom": "",
            "tune_with_ray": True,
            "retrain_model": True,
            "noise_dim": 100,
            "add_weather": True,
            "shuffle": True,
            "members": 5,
            "device": {"cpu": 1},
        }
        self.hostname_tsa = "tsa.example.com"
        self.hostname_nid = "nid.example.com"

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
        self.settings["retrain_model"] = True
        self.settings["tune_with_ray"] = True
        data_train, data_valid = xr.Dataset(), xr.Dataset()
        sha = "abc123"
        run_path = "my/run/path"
        train_and_evaluate_model(run_path, self.settings, data_train, data_valid, sha)
        mock_train_with_ray_tune.assert_called_once_with(
            run_path, self.settings, data_train, data_valid, sha
        )

        # Test retraining the model without Ray Tune
        self.settings["tune_with_ray"] = False
        train_and_evaluate_model(run_path, self.settings, data_train, data_valid, sha)
        mock_train_without_ray_tune.assert_called_once_with(
            self.settings, data_train, data_valid
        )

        # Test loading a pretrained model
        self.settings["retrain_model"] = False
        train_and_evaluate_model(run_path, self.settings, data_train, data_valid, sha)
        mock_load_pretrained_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
