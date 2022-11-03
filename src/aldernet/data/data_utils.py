"""Helper functions to import and pre-process zarr archives."""


def normalize_field(data, data_train):
    for i in range(data_train.shape[3]):
        center = data_train[:, :, :, i].mean(axis=(0, 1, 2), keepdims=True)
        scale = data_train[:, :, :, i].std(axis=(0, 1, 2), keepdims=True)
        data[:, :, :, i] = (data[:, :, :, i] - center) / scale
    return data
