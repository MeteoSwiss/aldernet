"""Helper functions to import and pre-process zarr archives."""

# Standard library
import math

# Third-party
import numpy as np
import tensorflow as tf  # type: ignore


class Params:
    """Retrieve selected weather parameters."""

    def __init__(self):
        """Define required data."""
        self.x = ["CORY"]
        self.y = ["ALNU"]
        self.weather = [
            "CORYctsum",
            "CORYfe",
            "CORYfr",
            "CORYrprec",
            "CORYsaisn",
            "CORYsdes",
            "cos_dayofyear",
            "cos_hourofday",
            "FIS",
            "HPBL",
            "HSURF",
            "QR",
            "P",
            "sin_dayofyear",
            "sin_hourofday",
            "TQC",
            "U",
            "V",
        ]
        self.combined = self.x + self.y + self.weather


class Stations:
    """Retrieve the measurement stations."""

    def __init__(self):
        """Define required data."""
        self.grids = {
            "grid_i": [
                525,
                511,
                651,
                677,
                420,
                472,
                456,
                603,
                614,
                570,
                636,
                479,
                540,
                590,
            ],
            "grid_j": [
                506,
                444,
                465,
                429,
                373,
                463,
                405,
                365,
                349,
                455,
                510,
                451,
                379,
                485,
            ],
        }
        self.name = [
            "PBS",
            "PBE",
            "PBU",
            "PDS",
            "PGE",
            "PCF",
            "PLS",
            "PLO",
            "PLU",
            "PLZ",
            "PMU",
            "PNE",
            "PVI",
            "PZH",
        ]


class Batcher(tf.keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, data, batch_size, add_weather, shuffle=True):
        """Initialize."""
        self.x = data[Params().x].to_array("var").transpose("valid_time", ..., "var")
        if add_weather:
            self.weather = (
                data[Params().weather]
                .to_array("var")
                .transpose("valid_time", ..., "var")
            )
        self.y = data[Params().y].to_array("var").transpose("valid_time", ..., "var")
        self.batch_size = batch_size
        self.add_weather = add_weather
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.add_weather:
            batch_weather = self.weather[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            return batch_x.values, batch_weather.values, batch_y.values
        else:
            return batch_x.values, batch_y.values

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        if self.shuffle is True:
            idx = np.random.permutation(len(self.x.values))
            self.x.values = self.x.values[idx]
            self.y.values = self.y.values[idx]
            if self.add_weather:
                self.weather.values = self.weather.values[idx]
            print("Data Reshuffled!", flush=True)
