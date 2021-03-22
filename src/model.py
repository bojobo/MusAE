from __future__ import print_function, division

import json
import logging as log
import os
from functools import partial

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
        phrase_size = midi_cfg.phrase_size

        regularisation_weight = training_cfg.regularisation_weight
        reconstruction_weight = training_cfg.reconstruction_weight

        log.info("Initialising encoder...")
        self.encoder = encoders.build_encoder_z()

        log.info("Initialising decoder...")
        self.decoder = decoders.build_decoder_z_flat()

        log.info("Initialising z discriminator...")
        z_discriminator = discriminators.build_gaussian_discriminator()

        # -------------------------------
        # Construct Computational Graph
        # for the Adversarial Autoencoder
        # -------------------------------
        log.info("Building reconstruction phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        z_discriminator.trainable = False

        x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X_recon")
        z_recon = self.encoder(x)
        y_piano, y_others = self.decoder(z_recon)

        reconstruction_phase = Model(
            inputs=x,
            outputs=[y_piano, y_others],
            name="autoencoder"
        )

        # -------------------------------
        # Construct Computational Graph
        #    for the z discriminator
        # -------------------------------
        log.info("Building z regularisation phase's computational graph...")
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
        log.info("Building generator regularisation phase's computational graph...")
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
        log.info("Building adversarial autoencoder's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        z_discriminator.trainable = True

        x = Input(shape=(phrase_size, midi_cfg.n_cropped_notes, midi_cfg.n_tracks), name="X")
        z_real = Input(shape=(model_cfg.z_length,), name="z")

        y_piano, y_others = reconstruction_phase(x)
        z_valid_real, z_valid_fake, z_valid_int, z_int = self.z_regularisation_phase([z_real, x])
        z_valid_gen = self.gen_regularisation_phase(x)

        self.adversarial_autoencoder = Model(
            inputs=[z_real, x],
            outputs=[
                y_piano, y_others,
                z_valid_real, z_valid_fake, z_valid_int,
                z_valid_gen
            ],
            name="adversarial_autoencoder"
        )

        y_pred = [y_piano, y_others]

        z_gp_loss = partial(h.gradient_penalty_loss, y_pred=y_pred, averaged_samples=z_int)
        z_gp_loss.__name__ = "gradient_penalty_z"

        self.adversarial_autoencoder.compile(
            loss=[
                "categorical_crossentropy", "categorical_crossentropy",
                h.wasserstein_loss, h.wasserstein_loss, z_gp_loss,
                h.wasserstein_loss
            ],
            loss_weights=[
                reconstruction_weight, reconstruction_weight,
                regularisation_weight, regularisation_weight, regularisation_weight * training_cfg.z_lambda,
                regularisation_weight
            ],
            optimizer=Adam(1e-5, clipnorm=1., clipvalue=.5),
            metrics=[
                "categorical_accuracy",
                h.output
            ]
        )

        print(self.adversarial_autoencoder.metrics_names)

    def train(self):
        epsilon_std = model_cfg.EncoderParams.epsilon_std
        regularisation_weight = training_cfg.regularisation_weight

        batch_x = [file for _, _, files in os.walk(cfg.Paths.x) for file in files]

        tr_batches, vl_batches = train_test_split(batch_x, shuffle=True, train_size=0.8)

        # storing losses over time
        tr_log = {
            "iteration": [],
            "AE_loss_piano": [],
            "AE_loss_others": [],
            "AE_loss_tot": [],
            "AE_accuracy_piano": [],
            "AE_accuracy_others": [],
            "AE_accuracy_tot": [],
            "z_score_real": [],
            "z_score_fake": [],
            "z_gradient_penalty": [],
            "gen_score": []
        }

        vl_log = {
            "epoch": [],
            "VL_AE_accuracy_piano": [],
            "VL_AE_accuracy_others": [],
            "VL_AE_accuracy_tot": []
        }

        pbc = 0
        pbc_tr = 0
        pbc_vl = 0
        annealing_first_stage = False
        annealing_second_stage = False
        annealing_third_stage = False
        for epoch in range(training_cfg.n_epochs):
            log.info("Epoch {} of {}".format(epoch + 1, training_cfg.n_epochs))
            log.info("Training on training set...")
            # train on the training set
            for batch in tr_batches:
                x = np.load(os.path.join(cfg.Paths.x, batch))
                y = np.load(os.path.join(cfg.Paths.y, batch))

                n_chunks = x.shape[0]

                # Adversarial ground truth (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z prior from mixture of gaussian
                z_real = np.random.normal(0, epsilon_std, (n_chunks, model_cfg.z_length))

                y_piano = y[:, :, :, 0]
                y_others = y[:, :, :, 1]

                aae_loss = self.adversarial_autoencoder.train_on_batch(
                    [z_real, x],
                    [
                        y_piano, y_others,
                        real_gt, fake_gt, dummy_gt,
                        real_gt
                    ]
                )

                tr_log["AE_loss_piano"].append(aae_loss[1])
                tr_log["AE_loss_others"].append(aae_loss[2])
                tr_log["AE_loss_tot"].append(np.array([aae_loss[1], aae_loss[2]]).mean())

                tr_log["AE_accuracy_piano"].append(aae_loss[7])
                tr_log["AE_accuracy_others"].append(aae_loss[9])
                tr_log["AE_accuracy_tot"].append(np.array([aae_loss[7], aae_loss[9]]).mean())

                tr_log["z_score_real"].append(aae_loss[12])
                tr_log["z_score_fake"].append(aae_loss[14])
                tr_log["z_gradient_penalty"].append(aae_loss[16])
                tr_log["gen_score"].append(aae_loss[18])

                if pbc_tr % 500 == 0:
                    log.info("Plotting stats...")
                    # print("Regularisation weight:", K.get_value(self.regularisation_weight))
                    h.plot(cfg.Paths.plots, tr_log)

                if pbc_tr % 5000 == 0:
                    log.info("Saving checkpoint...")
                    self.save_checkpoint(cfg.Paths.checkpoints, pbc_tr)

                # annealing the regularisation weight
                if pbc_tr > 1000 and not annealing_first_stage:
                    k.set_value(regularisation_weight, 0.0)
                    log.info("Regularisation weight annealed to ", k.get_value(regularisation_weight))
                    annealing_first_stage = True
                elif pbc_tr > 5000 and not annealing_second_stage:
                    k.set_value(regularisation_weight, 0.1)
                    log.info("Regularisation weight annealed to ", k.get_value(regularisation_weight))
                    annealing_second_stage = True
                elif pbc_tr > 10000 and not annealing_third_stage:
                    k.set_value(regularisation_weight, 0.2)
                    log.info("Regularisation weight annealed to ", k.get_value(regularisation_weight))
                    annealing_third_stage = True

                pbc += 1
                pbc_tr += 1
            # at the end of each epoch, we evaluate on the validation set
            log.info("Evaluating on validation set...")
            # evaluating on validation set

            vl_log_tmp = {
                "VL_AE_accuracy_piano": [],
                "VL_AE_accuracy_others": []
            }

            for batch in vl_batches:
                x = np.load(os.path.join(cfg.Paths.x, batch))
                y = np.load(os.path.join(cfg.Paths.y, batch))

                n_chunks = x.shape[0]

                # Adversarial ground truths (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z prior from mixture of gaussian
                z_real = np.random.normal(0, epsilon_std, (n_chunks, model_cfg.z_length))

                y_piano = y[:, :, :, 0]
                y_others = y[:, :, :, 1]
                aae_loss = self.adversarial_autoencoder.test_on_batch(
                    [z_real, x],
                    [
                        y_piano, y_others,
                        real_gt, fake_gt, dummy_gt,
                        real_gt
                    ]
                )

                vl_log_tmp["VL_AE_accuracy_piano"].append(aae_loss[7])
                vl_log_tmp["VL_AE_accuracy_others"].append(aae_loss[9])

                pbc += 1
                pbc_vl += 1

            log.info("Saving validation accuracy...")
            vl_log["epoch"].append(epoch)
            vl_log["VL_AE_accuracy_piano"].append(np.array(vl_log_tmp["VL_AE_accuracy_drums"]).mean())
            vl_log["VL_AE_accuracy_others"].append(np.array(vl_log_tmp["VL_AE_accuracy_bass"]).mean())
            vl_log["VL_AE_accuracy_tot"].append(
                np.array([vl_log["VL_AE_accuracy_piano"], vl_log["VL_AE_accuracy_others"]]).mean())

            with open(os.path.join(cfg.Paths.plots, "log.json"), 'w') as f:
                json.dump(str(vl_log), f)

            h.plot(cfg.Paths.plots, vl_log)

    def save_checkpoint(self, path, epoch):
        self.encoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_encoder.h5"))
        self.decoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_decoder.h5"))
