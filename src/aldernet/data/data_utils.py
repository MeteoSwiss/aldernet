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
                525 - 1,
                511 - 1,
                651 - 1,
                677 - 1,
                420 - 1,
                472 - 1,
                456 - 1,
                603 - 1,
                614 - 1,
                570 - 1,
                636 - 1,
                479 - 1,
                540 - 1,
                590 - 1,
            ],
            "grid_j": [
                506 - 1,
                444 - 1,
                465 - 1,
                429 - 1,
                373 - 1,
                463 - 1,
                405 - 1,
                365 - 1,
                349 - 1,
                455 - 1,
                510 - 1,
                451 - 1,
                379 - 1,
                485 - 1,
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
            idx = np.random.permutation(len(self.x))
            self.x = self.x[idx]
            self.y = self.y[idx]
            if self.add_weather:
                self.weather = self.weather[idx]
            print("Data Reshuffled!", flush=True)
