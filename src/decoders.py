from keras.engine.topology import Layer
from keras.layers import Input, Dense, CuDNNLSTM
from keras.layers import RepeatVector, TimeDistributed
from keras.models import Model

import config


# flat version
def build_decoder_z_flat():
    x_high_depth = config.model_params["decoder_params"]["X_high_depth"]
    x_high_size = config.model_params["decoder_params"]["X_high_size"]
    phrase_size = config.midi_params["phrase_size"]
    n_cropped_notes = config.midi_params["n_cropped_notes"]
    n_tracks = config.midi_params["n_tracks"]
    z_length = config.model_params["z_length"]

    z = Input(shape=(z_length,), name="z")
    decoder_inputs = z

    latent = z

    init_state = Dense(x_high_size, activation="tanh", name="hidden_state_init")(latent)

    out_x = []
    for t in range(n_tracks):
        h_x = RepeatVector(phrase_size, name=f"latent_repeat_{t}")(latent)
        for i in range(x_high_depth):
            h_x = CuDNNLSTM(
                x_high_size,
                return_sequences=True,
                # activation="tanh",
                name=f"high_encoder_{t}_{i}"
            )(h_x, initial_state=[init_state, init_state])

        out_x_t = TimeDistributed(
            Dense(n_cropped_notes, activation="softmax", name=f"project_out_{t}"),
            name=f"ts_project_{t}"
        )(h_x)

        out_x.append(out_x_t)

    decoder_outputs = out_x
    return Model(decoder_inputs, decoder_outputs, name="decoder")


class LowDecoder(Layer):
    def __init__(self, output_length, timestep_size, rec_depth, rec_size, **kwargs):
        self.output_length = output_length
        self.timestep_size = timestep_size
        self.rec_size = rec_size
        self.rec_depth = rec_depth
        super(LowDecoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.init_state_layer = Dense(self.rec_size, activation="linear", name="low_hidden_state_init")
        self.repeat_layer = RepeatVector(self.output_length, name=f"latent_repeat")

        self.lstm_layers = []
        for i in range(self.rec_depth):
            self.lstm_layers.append(CuDNNLSTM(self.rec_size, return_sequences=True, name=f"low_decoder_{i}"))

        self.project_layer = TimeDistributed(
            Dense(self.timestep_size, activation="softmax", name="project_X"),
            name="ts_project_X"
        )

        super(LowDecoder, self).build(input_shape)  # Be sure to call this at the end

    def call(self, embedding):

        # print("lowdec - embedding:", embedding.shape)

        init_state = self.init_state_layer(embedding)
        h_x = self.repeat_layer(embedding)

        # print("lowdec - h_X:", h_X.shape)

        for i in range(self.rec_depth):
            h_x = self.lstm_layers[i](h_x, initial_state=[init_state, init_state])

        # print("lowdec - h_X:", h_X.shape)

        out_x = self.project_layer(h_x)

        # print("lowdec - out_X:", out_X.shape)

        return out_x

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        batch_size = input_shape[0]
        return batch_size, self.output_length, self.timestep_size
