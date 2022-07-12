import ml_collections


def get_mixer_config():
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patch_size = 32
    config.hidden_dim = 512
    config.num_blocks = 8
    config.tokens_mlp_dim = 256
    config.channels_mlp_dim = 2048
    return config
