import logging as log
from functools import partial
from glob import glob
from statistics import mean

import numpy as np
from keras import backend as k
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import config as cfg
import decoders
import discriminators
import encoders
import helper as h
import midi_cfg
import model_cfg
import training_cfg

log.getLogger(__name__)


class MusAE_GM:
    def __init__(self):
        log.info("Initializing MusAE...")
        phrase_size = midi_cfg.phrase_size

        self.regularisation_weight = k.variable(value=0.0, name="reg_weight")
        reconstruction_weight = training_cfg.reconstruction_weight

        log.info(" - Initialising encoder...")
        self.encoder = encoders.EncoderZ()

        log.info(" - Initialising decoder...")
        self.decoder = decoders.DecoderZFlat()

        log.info(" - Initialising z discriminator...")
        z_discriminator = discriminators.GaussianDiscriminator()

        # -------------------------------
        # Construct Computational Graph
        # for the Adversarial Autoencoder
        # -------------------------------
        log.info(" - Building reconstruction phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        z_discriminator.trainable = False

        x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X_recon")
        z_recon = self.encoder(x)
        decoder_output = self.decoder(z_recon)

        reconstruction_phase = Model(
            inputs=x,
            outputs=decoder_output,
            name="autoencoder"
        )

        # -------------------------------
        # Construct Computational Graph
        #    for the z discriminator
        # -------------------------------
        log.info(" - Building z regularisation phase's computational graph...")
        self.encoder.trainable = False
        self.decoder.trainable = False
        z_discriminator.trainable = True

        x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X_z_reg")

        z_real = Input(shape=(model_cfg.z_length,), name="z_reg")
        z_fake = self.encoder(x)

        z_int = h.RandomWeightedAverage(name="weighted_avg_z")([z_real, z_fake])

        z_valid_real = z_discriminator(z_real)
        z_valid_fake = z_discriminator(z_fake)
        z_valid_int = z_discriminator(z_int)

        self.z_regularisation_phase = Model(
            [z_real, x],
            [z_valid_real, z_valid_fake, z_valid_int, z_int],
            name="z_regularisation_phase"
        )

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        log.info(" - Building generator regularisation phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = False
        z_discriminator.trainable = False

        x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X_gen_reg")

        z_gen = self.encoder(x)
        z_valid_gen = z_discriminator(z_gen)

        self.gen_regularisation_phase = Model(
            inputs=x,
            outputs=z_valid_gen,
            name="gen_regularisation_phase"
        )

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        log.info(" - Building adversarial autoencoder's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        z_discriminator.trainable = True

        x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X")
        z_real = Input(shape=(model_cfg.z_length,), name="z")

        reconstruction_output = reconstruction_phase(x)
        z_valid_real, z_valid_fake, z_valid_int, z_int = self.z_regularisation_phase([z_real, x])
        z_valid_gen = self.gen_regularisation_phase(x)

        outputs = []
        outputs.extend(reconstruction_output)
        outputs.extend([z_valid_real, z_valid_fake, z_valid_int, z_valid_gen])
        self.adversarial_autoencoder = Model(
            inputs=[z_real, x],
            outputs=outputs,
            name="adversarial_autoencoder"
        )

        z_gp_loss = partial(h.gradient_penalty_loss, y_pred=reconstruction_output, averaged_samples=z_int)
        z_gp_loss.__name__ = "gradient_penalty_z"

        loss = []
        loss_weights = []
        for _ in range(midi_cfg.n_tracks):
            loss.append("categorical_crossentropy")
            loss_weights.append(reconstruction_weight)
        loss.extend([h.wasserstein_loss, h.wasserstein_loss, z_gp_loss, h.wasserstein_loss])
        loss_weights.extend([self.regularisation_weight, self.regularisation_weight,
                             self.regularisation_weight * training_cfg.z_lambda,
                             self.regularisation_weight])
        self.adversarial_autoencoder.compile(
            loss=loss,
            loss_weights=loss_weights,
            optimizer=Adam(1e-5, clipnorm=1., clipvalue=.5),
            metrics=[
                "categorical_accuracy",
                h.output
            ]
        )
        log.info(" - Done!")

    def train(self):
        batches = glob(cfg.Paths.musae + "/**/batch[0-9]*.npz", recursive=True)
        log.info("Found {} batches.".format(len(batches)))
        tr_batches, vl_batches = train_test_split(batches, shuffle=True, train_size=0.8)

        log.info("Batches list: {}".format(batches))
        batch = np.load(batches[0])
        log.info("x shape = {}".format(batch['x'].shape))
        log.info("y shape = {}".format(batch['y'].shape))

        # storing losses over time
        tr_log = {}
        for i in range(1, midi_cfg.n_tracks + 1):
            tr_log[f"TR_Loss_Track_{i}"] = []
        for key in ["TR_Loss_Total", "Z_Score_Real", "Z_Score_Fake", "Z_Score_Penalty", "Gen_Score"]:
            tr_log[key] = []
        tr_log["TR_Loss_Total"] = []

        vl_log = {}
        for i in range(1, midi_cfg.n_tracks + 1):
            vl_log[f"VL_Accuracy_Track_{i}"] = []
        vl_log["VL_Accuracy_Total"] = []

        best_total_vl_accuracy = -1
        for epoch in range(1, training_cfg.n_epochs + 1):
            log.info("Epoch {} of {}".format(epoch, training_cfg.n_epochs))
            if (epoch % int(training_cfg.n_epochs / 10)) == 0:
                old = k.get_value(self.regularisation_weight)
                new = old + 0.1
                k.set_value(self.regularisation_weight, new)
                log.info(" - Regularisation weight annealed to {}".format(k.get_value(self.regularisation_weight)))
                log.info(" - Plotting metrics graphs")
                h.plot(tr_log)
                h.plot(vl_log)

            # train on the training set
            log.info(" - Training on training set...")
            training_losses = self._run(tr_batches, False)
            for training_loss in training_losses:
                sum_loss = 0
                for i in range(1, midi_cfg.n_tracks + 1):
                    loss = training_loss[i]
                    sum_loss += loss
                    tr_log[f"TR_Loss_Track_{i}"].append(loss)
                tr_log["TR_Loss_Total"].append(sum_loss / midi_cfg.n_tracks)

                for i, key in enumerate(["Z_Score_Real", "Z_Score_Fake", "Z_Score_Penalty", "Gen_Score"]):
                    tr_log[key].append(training_loss[6 + 2 * midi_cfg.n_tracks + (i * 2)])

            # at the end of each epoch, we evaluate on the validation set
            log.info(" - Evaluating on validation set...")
            validation_losses = self._run(vl_batches, True)
            for validation_loss in validation_losses:
                sum_mean_loss = 0
                vl_log_tmp = {}
                for i in range(1, midi_cfg.n_tracks + 1):
                    vl_log_tmp[f"VL_Accuracy_Track_{i}"] = []
                for i in range(1, midi_cfg.n_tracks + 1):
                    # Collect all accuracy values
                    loss = validation_loss[5 + midi_cfg.n_tracks + (i * 2)]
                    vl_log_tmp[f"VL_Accuracy_Track_{i}"].append(loss)
                for i in range(1, midi_cfg.n_tracks + 1):
                    # Calculate the mean accuracy for each track
                    mean_loss = mean(vl_log_tmp[f"VL_Accuracy_Track_{i}"])
                    sum_mean_loss += mean_loss
                    vl_log[f"VL_Accuracy_Track_{i}"].append(mean_loss)
                # Calculate the overall accuracy
                vl_log["VL_Accuracy_Total"].append(sum_mean_loss / midi_cfg.n_tracks)

                if vl_log["VL_Accuracy_Total"][-1] > best_total_vl_accuracy:
                    best_total_vl_accuracy = vl_log["VL_Accuracy_Total"][-1]
                    self.save_checkpoint()

    def _run(self, batches, validation: bool) -> list:
        losses = []
        for batch in batches:
            batch = np.load(batch)

            x = batch['x']
            y = batch['y']
            n_chunks = x.shape[0]

            # Adversarial ground truth (wasserstein)
            real_gt = -np.ones((n_chunks, 1))
            fake_gt = np.ones((n_chunks, 1))
            dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

            # draw z prior from mixture of gaussian
            z_real = np.random.normal(0, model_cfg.EncoderParams.epsilon_std, (n_chunks, model_cfg.z_length))

            aae_y = []
            for i in range(midi_cfg.n_tracks):
                aae_y.append(y[:, :, :, i])
            aae_y.extend([real_gt, fake_gt, dummy_gt, real_gt])

            if not validation:
                aae_loss = self.adversarial_autoencoder.train_on_batch([z_real, x], aae_y)
            else:
                aae_loss = self.adversarial_autoencoder.test_on_batch([z_real, x], aae_y)
            losses.append(aae_loss)
        return losses

    def save_checkpoint(self):
        self.encoder.save(cfg.Resources.best_encoder)
        self.decoder.save(cfg.Resources.best_decoder)
