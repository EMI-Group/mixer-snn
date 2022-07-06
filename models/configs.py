import ml_collections


def get_mixer_b16_config():
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_dim = 768
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    return config
