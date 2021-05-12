from keras.layers import Input, Dense, CuDNNLSTM
from keras.layers import RepeatVector, TimeDistributed
from keras.models import Model

import midi_cfg
import model_cfg


class DecoderZFlat(Model):
    def __init__(self, *args, **kwargs):
        z = Input(shape=(model_cfg.z_length,), name="z")
        init_state = Dense(model_cfg.DecoderParams.x_high_size, activation="tanh", name="hidden_state_init")(z)

        out_x = []
        for i in range(midi_cfg.n_tracks):
            h_x = RepeatVector(midi_cfg.phrase_size, name=f"latent_repeat_{i}")(z)
            for j in range(model_cfg.DecoderParams.x_high_depth):
                h_x = CuDNNLSTM(
                    model_cfg.DecoderParams.x_high_size,
                    return_sequences=True,
                    name=f"high_encoder_{i}_{j}"
                )(h_x, initial_state=[init_state, init_state])

            out_x.append(
                TimeDistributed(
                    Dense(midi_cfg.n_cropped_notes, activation="softmax", name=f"project_out_{i}"),
                    name=f"ts_project_{i}"
                )(h_x)
            )

        super().__init__(z, out_x, name='decoder_z_flat')
