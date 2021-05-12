from keras.layers import Input, Dense
from keras.models import Model

import model_cfg


class GaussianDiscriminator(Model):
    def __init__(self, *args, **kwargs):
        z = Input(shape=(model_cfg.z_length,), name="z")
        h = z
        for i in range(model_cfg.ZDiscriminatorParams.fc_depth):
            h = Dense(model_cfg.ZDiscriminatorParams.fc_size, activation='tanh', name=f"fc_{i}")(h)
        out = Dense(1, activation='linear', name='validity')(h)
        super().__init__(z, out, name="z_discriminator")
