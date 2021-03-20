from keras.layers import Input, Dense
from keras.models import Model

import model_cfg


def build_gaussian_discriminator():
    fc_depth = model_cfg.ZDiscriminatorParams.fc_depth
    fc_size = model_cfg.ZDiscriminatorParams.fc_size
    z_length = model_cfg.z_length

    z = Input(shape=(z_length,), name="z")

    # fully connected layers
    h = z
    for i in range(fc_depth):
        h = Dense(fc_size, activation="tanh", name=f"fc_{i}")(h)
    # h = BatchNormalization(name=f"batchnorm_fc_{l}")(h)

    out = Dense(1, activation="linear", name="validity")(h)

    return Model(z, out, name="z_discriminator")
