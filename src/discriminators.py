from keras.layers import Input, Dense, Concatenate
from keras.models import Model

import config


def build_gaussian_discriminator():
    fc_depth = config.model_params["z_discriminator_params"]["fc_depth"]
    fc_size = config.model_params["z_discriminator_params"]["fc_size"]
    z_length = config.model_params["z_length"]

    z = Input(shape=(z_length,), name="z")

    # fully connected layers
    h = z
    for i in range(fc_depth):
        h = Dense(fc_size, activation="tanh", name=f"fc_{i}")(h)
    # h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

    out = Dense(1, activation="linear", name="validity")(h)

    return Model(z, out, name="z_discriminator")


def build_bernoulli_discriminator():
    fc_depth = config.model_params["s_discriminator_params"]["fc_depth"]
    fc_size = config.model_params["s_discriminator_params"]["fc_size"]
    s_length = config.model_params["s_length"]

    s = Input(shape=(s_length,), name="s")

    # fully connected layers
    h = s
    for i in range(fc_depth):
        h = Dense(fc_size, activation="tanh", name=f"fc_{i}")(h)
    # h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

    out = Dense(1, activation="linear", name="validity")(h)

    return Model(s, out, name="s_discriminator")


def build_gaussian_mixture_discriminator():
    fc_depth = config.model_params["s_discriminator_params"]["fc_depth"]
    fc_size = config.model_params["s_discriminator_params"]["fc_size"]
    z_length = config.model_params["z_length"]
    s_length = config.model_params["s_length"]

    z = Input(shape=(z_length,), name="z")
    y = Input(shape=(s_length,), name="y")

    h = Concatenate(axis=-1, name="concat")([z, y])

    # fully connected layers
    for i in range(fc_depth):
        h = Dense(fc_size, activation="tanh", name=f"fc_{i}")(h)
    # h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

    out = Dense(1, activation="linear", name="validity")(h)

    return Model([z, y], out, name="s_discriminator")
