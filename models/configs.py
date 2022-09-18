import ml_collections


def get_mixer_sparse_tiny_config():
    config = ml_collections.ConfigDict()
    config.in_channels = 3
    config.dim = 80
    config.alpha = 3
    config.patch_size = 4
    config.image_size = 224
    config.depths = [2, 8, 14, 2]
    return config
