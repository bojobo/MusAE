from keras import backend as k
from keras.layers import Input, Dense, Bidirectional, CuDNNLSTM
from keras.layers import Lambda
from keras.layers import Reshape
from keras.models import Model

import midi_cfg
import model_cfg


def build_encoder_z():
    x_depth = model_cfg.EncoderParams.x_depth
    x_size = model_cfg.EncoderParams.x_size
    epsilon_std = model_cfg.EncoderParams.epsilon_std
    phrase_size = midi_cfg.phrase_size
    z_length = model_cfg.z_length

    x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X")
    encoder_inputs = x

    # X encoder
    h_x = Reshape((phrase_size, midi_cfg.n_tracks * midi_cfg.n_cropped_notes), name="reshape_X")(x)
    for i in range(x_depth - 1):
        h_x = Bidirectional(
            CuDNNLSTM(x_size, return_sequences=True, name=f"rec_X_{i}"),
            merge_mode="concat", name=f"bidirectional_X_{i}"
        )(h_x)
    # h_X = BatchNormalization(name=f"batchnorm_X_{l}")(h_X)

    h = Bidirectional(
        CuDNNLSTM(x_size, return_sequences=False, name=f"rec_X_{x_depth - 1}"),
        merge_mode="concat", name=f"bidirectional_X_{x_depth - 1}"
    )(h_x)
    # h = BatchNormalization(name=f"batchnorm_X_{X_depth}")(h_X)

    # s = Dense(s_length, name="s", activation="sigmoid")(h)

    # reparameterisation trick
    z_mean = Dense(z_length, name='mu', activation='linear')(h)
    z_log_var = Dense(z_length, name='sigma', activation='linear')(h)

    # sampling
    def sampling(args):
        z_mean_, z_log_var_ = args
        batch_size = k.shape(z_mean_)[0]
        epsilon = k.random_normal(shape=(batch_size, z_length), mean=0., stddev=epsilon_std)
        return z_mean_ + k.exp(z_log_var_ / 2) * epsilon

    z = Lambda(sampling, output_shape=(z_length,), name='z_sampling')([z_mean, z_log_var])

    encoder_outputs = z

    return Model(encoder_inputs, encoder_outputs, name="encoder")
