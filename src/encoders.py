from keras import backend as k
from keras.layers import Input, Dense, Bidirectional, CuDNNLSTM
from keras.layers import Lambda
from keras.layers import Reshape
from keras.models import Model

import midi_cfg
import model_cfg


class EncoderZ(Model):
    def __init__(self):
        x = Input(shape=(midi_cfg.phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X")

        h_x = Reshape((midi_cfg.phrase_size, midi_cfg.n_tracks * midi_cfg.n_cropped_notes), name="reshape_X")(x)
        for i in range(model_cfg.EncoderParams.x_depth - 1):
            h_x = Bidirectional(
                CuDNNLSTM(model_cfg.EncoderParams.x_size, return_sequences=True, name=f"rec_X_{i}"),
                merge_mode="concat", name=f"bidirectional_X_{i}"
            )(h_x)

        h = Bidirectional(
            CuDNNLSTM(model_cfg.EncoderParams.x_size, return_sequences=False,
                      name=f"rec_X_{model_cfg.EncoderParams.x_depth - 1}"),
            merge_mode="concat", name=f"bidirectional_X_{model_cfg.EncoderParams.x_depth - 1}"
        )(h_x)

        # reparameterisation trick
        z_mean = Dense(model_cfg.z_length, name='mu', activation='linear')(h)
        z_log_var = Dense(model_cfg.z_length, name='sigma', activation='linear')(h)

        # sampling
        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = k.shape(z_mean_)[0]
            epsilon = k.random_normal(shape=(batch_size, model_cfg.z_length), mean=0.,
                                      stddev=model_cfg.EncoderParams.epsilon_std)
            return z_mean_ + k.exp(z_log_var_ / 2) * epsilon

        z = Lambda(sampling, output_shape=(model_cfg.z_length,), name='z_sampling')([z_mean, z_log_var])

        super().__init__(x, z, name="encoder_z")
