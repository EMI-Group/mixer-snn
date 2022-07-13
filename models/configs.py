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


def get_mixer_v1_config():
    config = ml_collections.ConfigDict()
    config.name = 'mixer_v1'
    config.patch_size = 4
    config.hidden_dim = 256
    config.num_blocks = 8
    return config


def get_multi_stage_model_config():
    config = ml_collections.ConfigDict()
    config.name = 'multi_stage_model'
    config.height = 224
    config.width = 224
    config.in_channels = 3
    config.hidden_dim = 64
    config.patch_size = 4
    config.depth = [2, 2, 2]
    config.num_classes = 10
    return config
