import ml_collections


def get_mixer_conv_encode_config():
    config = ml_collections.ConfigDict()
    config.name = 'mixer_conv_encode'
    config.img_size = 224
    config.patch_size = 32
    config.encode_dim = 32
    config.hidden_dim = 768
    config.token_hidden_dim = 256
    config.channel_hidden_dim = 2048
    config.num_blocks = 2
    config.voting_num = 10
    return config


def get_mixer_mlp_encode_config():
    config = ml_collections.ConfigDict()
    config.name = 'mixer_mlp_encode'
    config.img_size = 224
    config.patch_size = 32
    config.hidden_dim = 768
    config.token_hidden_dim = 384
    config.channel_hidden_dim = 3072
    config.num_blocks = 12
    return config
