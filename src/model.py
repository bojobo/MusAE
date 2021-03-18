from __future__ import print_function, division

import json
import logging as log
import os
from functools import partial

import numpy as np
from keras import backend as k
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split

import config as cfg
import decoders
import discriminators
import encoders
import helper as h

log.getLogger("MusAE_GM")


class MusAE_GM:
    def __init__(self):
        self.name = "MusAE_GM"
        self.s_length = cfg.model_params["s_length"]
        self.z_length = cfg.model_params["z_length"]

        self.phrase_size = cfg.midi_params["phrase_size"]
        self.n_cropped_notes = cfg.midi_params["n_cropped_notes"]
        self.n_tracks = cfg.midi_params["n_tracks"]

        self.regularisation_weight = cfg.training_params["regularisation_weight"]
        self.reconstruction_weight = cfg.training_params["reconstruction_weight"]
        self.z_lambda = cfg.training_params["z_lambda"]
        self.aae_optim = cfg.training_params["aae_optim"]
        self.n_epochs = cfg.training_params["n_epochs"]

        log.info("Initialising encoder...")
        self.encoder = encoders.build_encoder_z()

        log.info("Initialising decoder...")
        self.decoder = decoders.build_decoder_z_flat()

        log.info("Initialising z discriminator...")
        self.z_discriminator = discriminators.build_gaussian_discriminator()

        # -------------------------------
        # Construct Computational Graph
        # for the Adversarial Autoencoder
        # -------------------------------
        log.info("Building reconstruction phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_recon")
        z_recon = self.encoder(x)
        y_piano, y_strings, y_ensemble, y_others = self.decoder(z_recon)

        self.reconstruction_phase = Model(
            inputs=x,
            outputs=[y_piano, y_strings, y_ensemble, y_others],
            name="autoencoder"
        )

        # -------------------------------
        # Construct Computational Graph
        #    for the z discriminator
        # -------------------------------
        log.info("Building z regularisation phase's computational graph...")
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.z_discriminator.trainable = True

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_z_reg")

        z_real = Input(shape=(self.z_length,), name="z_reg")
        z_fake = self.encoder(x)

        z_int = h.RandomWeightedAverage(name="weighted_avg_z")([z_real, z_fake])

        z_valid_real = self.z_discriminator(z_real)
        z_valid_fake = self.z_discriminator(z_fake)
        z_valid_int = self.z_discriminator(z_int)

        self.z_regularisation_phase = Model(
            [z_real, x],
            [z_valid_real, z_valid_fake, z_valid_int, z_int],
            name="z_regularisation_phase"
        )

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        log.info("Building generator regularisation phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.z_discriminator.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_gen_reg")

        z_gen = self.encoder(x)
        z_valid_gen = self.z_discriminator(z_gen)

        self.gen_regularisation_phase = Model(
            inputs=x,
            outputs=z_valid_gen,
            name="gen_regularisation_phase"
        )

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        log.info("Building adversarial autoencoder's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = True

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
        z_real = Input(shape=(self.z_length,), name="z")

        y_piano, y_strings, y_ensemble, y_others = self.reconstruction_phase(x)
        z_valid_real, z_valid_fake, z_valid_int, z_int = self.z_regularisation_phase([z_real, x])
        z_valid_gen = self.gen_regularisation_phase(x)

        self.adversarial_autoencoder = Model(
            inputs=[z_real, x],
            outputs=[
                y_piano, y_strings, y_ensemble, y_others,
                z_valid_real, z_valid_fake, z_valid_int,
                z_valid_gen
            ],
            name="adversarial_autoencoder"
        )

        y_pred = [y_piano, y_strings, y_ensemble, y_others]

        z_gp_loss = partial(h.gradient_penalty_loss, y_pred=y_pred, averaged_samples=z_int)
        z_gp_loss.__name__ = "gradient_penalty_z"

        self.adversarial_autoencoder.compile(
            loss=[
                "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy",
                "categorical_crossentropy",
                h.wasserstein_loss, h.wasserstein_loss, z_gp_loss,
                h.wasserstein_loss
            ],
            loss_weights=[
                self.reconstruction_weight, self.reconstruction_weight, self.reconstruction_weight,
                self.reconstruction_weight,
                self.regularisation_weight, self.regularisation_weight, self.regularisation_weight * self.z_lambda,
                self.regularisation_weight
            ],
            optimizer=self.aae_optim,
            metrics=[
                "categorical_accuracy",
                h.output
            ]
        )

    def train(self):
        epsilon_std = cfg.model_params["encoder_params"]["epsilon_std"]
        # create checkpoint and plots folder
        paths = {
            "checkpoints": os.path.join(cfg.checkpoints_path, self.name),
            "plots": os.path.join(cfg.plots_path, self.name),
        }
        for key in paths:
            h.create_dirs(paths[key])

        # Remove empty batches
        # training_batches = [batch for batch in training_batches if batch]
        # test_batches = [batch for batch in test_batches if batch]

        batch_x = [file for _, _, files in os.walk(cfg.x_path) for file in files]

        tr_batches, vl_batches = train_test_split(batch_x, shuffle=True, train_size=0.8)

        # storing losses over time
        tr_log = {
            "iteration": [],
            "AE_loss_drums": [],
            "AE_loss_bass": [],
            "AE_loss_guitar": [],
            "AE_loss_strings": [],
            "AE_loss_tot": [],
            "AE_accuracy_drums": [],
            "AE_accuracy_bass": [],
            "AE_accuracy_guitar": [],
            "AE_accuracy_strings": [],
            "AE_accuracy_tot": [],
            "z_score_real": [],
            "z_score_fake": [],
            "z_gradient_penalty": [],
            "gen_score": []
        }

        vl_log = {
            "epoch": [],
            "VL_AE_accuracy_drums": [],
            "VL_AE_accuracy_bass": [],
            "VL_AE_accuracy_guitar": [],
            "VL_AE_accuracy_strings": [],
            "VL_AE_accuracy_tot": []
        }

        pbc = 0
        pbc_tr = 0
        pbc_vl = 0
        annealing_first_stage = False
        annealing_second_stage = False
        annealing_third_stage = False
        for epoch in range(self.n_epochs):
            log.info("Epoch", epoch + 1, "of", self.n_epochs)
            log.info("Training on training set...")
            # train on the training set
            for batch in tr_batches:
                x = np.load(os.path.join(cfg.x_path, batch))
                y = np.load(os.path.join(cfg.y_path, batch))

                n_chunks = x.shape[0]

                # Adversarial ground truth (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z prior from mixture of gaussian
                z_real = np.random.normal(0, epsilon_std, (n_chunks, self.z_length))

                y_piano = y[:, :, :, 0]
                y_strings = y[:, :, :, 1]
                y_ensemble = y[:, :, :, 2]
                y_others = y[:, :, :, 3]

                aae_loss = self.adversarial_autoencoder.train_on_batch(
                    [z_real, x],
                    [
                        y_piano, y_strings, y_ensemble, y_others,
                        real_gt, fake_gt, dummy_gt,
                        real_gt
                    ]
                )

                tr_log["AE_loss_drums"].append(aae_loss[1])
                tr_log["AE_loss_bass"].append(aae_loss[2])
                tr_log["AE_loss_guitar"].append(aae_loss[3])
                tr_log["AE_loss_strings"].append(aae_loss[4])
                tr_log["AE_loss_tot"].append(np.array([aae_loss[1], aae_loss[2], aae_loss[3], aae_loss[4]]).mean())

                tr_log["AE_accuracy_drums"].append(aae_loss[9])
                tr_log["AE_accuracy_bass"].append(aae_loss[11])
                tr_log["AE_accuracy_guitar"].append(aae_loss[13])
                tr_log["AE_accuracy_strings"].append(aae_loss[15])
                tr_log["AE_accuracy_tot"].append(
                    np.array([aae_loss[9], aae_loss[11], aae_loss[13], aae_loss[15]]).mean())

                tr_log["z_score_real"].append(aae_loss[18])
                tr_log["z_score_fake"].append(aae_loss[20])
                tr_log["z_gradient_penalty"].append(aae_loss[22])
                tr_log["gen_score"].append(aae_loss[24])

                if pbc_tr % 500 == 0:
                    log.info("Plotting stats...")
                    # print("Regularisation weight:", K.get_value(self.regularisation_weight))
                    h.plot(paths["plots"], tr_log)

                if pbc_tr % 5000 == 0:
                    log.info("Saving checkpoint...")
                    self.save_checkpoint(paths["checkpoints"], pbc_tr)

                # annealing the regularisation weight
                if pbc_tr > 1000 and not annealing_first_stage:
                    k.set_value(self.regularisation_weight, 0.0)
                    log.info("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_first_stage = True
                elif pbc_tr > 5000 and not annealing_second_stage:
                    k.set_value(self.regularisation_weight, 0.1)
                    log.info("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_second_stage = True
                elif pbc_tr > 10000 and not annealing_third_stage:
                    k.set_value(self.regularisation_weight, 0.2)
                    log.info("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_third_stage = True

                pbc += 1
                pbc_tr += 1
            # at the end of each epoch, we evaluate on the validation set
            log.info("Evaluating on validation set...")
            # evaluating on validation set

            vl_log_tmp = {
                "VL_AE_accuracy_drums": [],
                "VL_AE_accuracy_bass": [],
                "VL_AE_accuracy_guitar": [],
                "VL_AE_accuracy_strings": [],
            }

            for batch in vl_batches:
                x = np.load(os.path.join(cfg.x_path, batch))
                y = np.load(os.path.join(cfg.y_path, batch))

                n_chunks = x.shape[0]

                # Adversarial ground truths (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z prior from mixture of gaussian
                z_real = np.random.normal(0, epsilon_std, (n_chunks, self.z_length))

                y_piano = y[:, :, :, 0]
                y_strings = y[:, :, :, 1]
                y_ensemble = y[:, :, :, 2]
                y_others = y[:, :, :, 3]
                aae_loss = self.adversarial_autoencoder.test_on_batch(
                    [z_real, x],
                    [
                        y_piano, y_strings, y_ensemble, y_others,
                        real_gt, fake_gt, dummy_gt,
                        real_gt
                    ]
                )

                vl_log_tmp["VL_AE_accuracy_drums"].append(aae_loss[9])
                vl_log_tmp["VL_AE_accuracy_bass"].append(aae_loss[11])
                vl_log_tmp["VL_AE_accuracy_guitar"].append(aae_loss[13])
                vl_log_tmp["VL_AE_accuracy_strings"].append(aae_loss[15])

                pbc += 1
                pbc_vl += 1

            log.info("Saving validation accuracy...")
            vl_log["epoch"].append(epoch)
            vl_log["VL_AE_accuracy_drums"].append(np.array(vl_log_tmp["VL_AE_accuracy_drums"]).mean())
            vl_log["VL_AE_accuracy_bass"].append(np.array(vl_log_tmp["VL_AE_accuracy_bass"]).mean())
            vl_log["VL_AE_accuracy_guitar"].append(np.array(vl_log_tmp["VL_AE_accuracy_guitar"]).mean())
            vl_log["VL_AE_accuracy_strings"].append(np.array(vl_log_tmp["VL_AE_accuracy_strings"]).mean())
            vl_log["VL_AE_accuracy_tot"].append(np.array(
                [vl_log["VL_AE_accuracy_drums"], vl_log["VL_AE_accuracy_bass"], vl_log["VL_AE_accuracy_guitar"],
                 vl_log["VL_AE_accuracy_strings"]]).mean())

            with open(os.path.join(paths["plots"], "log.json"), 'w') as f:
                json.dump(str(vl_log), f)

            h.plot(paths["plots"], vl_log)

    def save_checkpoint(self, path, epoch):
        self.encoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_encoder.h5"))
        self.decoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_decoder.h5"))
