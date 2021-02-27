from __future__ import print_function, division

import json
import logging as log
import os
import pprint
import threading
from functools import partial
from queue import Queue
from typing import List

import numpy as np
from keras import backend as k
from keras.layers import Concatenate
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model

import config as cfg
import decoders
import discriminators
import encoders
import helper as h

log.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


class MusAE:
    def __init__(self):
        # setting params as class attributes
        self.name = "MusAE"
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
        self.s_lambda = cfg.training_params["s_lambda"]
        self.supervised_weight = cfg.training_params["supervised_weight"]
        self.infomax_weight = cfg.training_params["infomax_weight"]

        log.info("Initialising encoder...")
        self.encoder = encoders.build_encoder_sz()

        log.info("Initialising decoder...")
        self.decoder = decoders.build_decoder_sz_flat()

        log.info("Initialising z discriminator...")
        self.z_discriminator = discriminators.build_gaussian_discriminator()

        log.info("Initialising s discriminator...")
        self.s_discriminator = discriminators.build_bernoulli_discriminator()

        # print("Initialising infomax network...")
        # self.infomax_net = discriminators.build_infomax_network()

        path = os.path.join(cfg.plots_path, self.name, "models")
        h.create_dirs(path)

        log.info("Saving model plots..")
        plot_model(self.encoder, os.path.join(path, "encoder.png"), show_shapes=True)
        plot_model(self.decoder, os.path.join(path, "decoder.png"), show_shapes=True)
        plot_model(self.z_discriminator, os.path.join(path, "z_discriminator.png"), show_shapes=True)
        plot_model(self.s_discriminator, os.path.join(path, "s_discriminator.png"), show_shapes=True)
        # plot_model(self.infomax_net, os.path.join(path, "infomax_net.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the Adversarial Autoencoder
        # -------------------------------
        log.info("Building reconstruction phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = False
        self.s_discriminator.trainable = False
        # self.infomax_net.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_recon")
        s_recon, z_recon = self.encoder(x)
        y_drums, y_bass, y_guitar, y_strings = self.decoder([s_recon, z_recon])

        self.reconstruction_phase = Model(
            inputs=x,
            outputs=[y_drums, y_bass, y_guitar, y_strings],
            name="autoencoder"
        )
        plot_model(self.reconstruction_phase, os.path.join(path, "reconstruction_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        #    for the z discriminator
        # -------------------------------
        log.info("Building z regularisation phase's computational graph...")
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.z_discriminator.trainable = True
        self.s_discriminator.trainable = False
        # self.infomax_net.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_z_reg")
        z_real = Input(shape=(self.z_length,), name="z_reg")
        _, z_fake = self.encoder(x)
        z_int = h.RandomWeightedAverage(name="weighted_avg_z")([z_real, z_fake])

        z_valid_real = self.z_discriminator(z_real)
        z_valid_fake = self.z_discriminator(z_fake)
        z_valid_int = self.z_discriminator(z_int)

        self.z_regularisation_phase = Model(
            [z_real, x],
            [z_valid_real, z_valid_fake, z_valid_int, z_int],
            name="z_regularisation_phase"
        )
        plot_model(self.z_regularisation_phase, os.path.join(path, "z_regularisation_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        #    for the s discriminator
        # -------------------------------
        log.info("Building s regularisation phase's computational graph...")
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.z_discriminator.trainable = False
        self.s_discriminator.trainable = True
        # self.infomax_net.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_s_reg")
        s_real = Input(shape=(self.s_length,), name="s_reg")

        s_fake, _ = self.encoder(x)
        s_int = h.RandomWeightedAverage(name="weighted_avg_s")([s_real, s_fake])

        s_valid_real = self.s_discriminator(s_real)
        s_valid_fake = self.s_discriminator(s_fake)
        s_valid_int = self.s_discriminator(s_int)

        self.s_regularisation_phase = Model(
            [s_real, x],
            [s_valid_real, s_valid_fake, s_valid_int, s_int],
            name="s_regularisation_phase"
        )
        plot_model(self.s_regularisation_phase, os.path.join(path, "s_regularisation_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        log.info("Building generator regularisation phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.z_discriminator.trainable = False
        self.s_discriminator.trainable = False
        # self.infomax_net.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_gen_reg")

        s_gen, z_gen = self.encoder(x)

        z_valid_gen = self.z_discriminator(z_gen)
        s_valid_gen = self.s_discriminator(s_gen)

        self.gen_regularisation_phase = Model(
            inputs=x,
            outputs=[s_valid_gen, z_valid_gen],
            name="gen_regularisation_phase"
        )
        plot_model(self.gen_regularisation_phase, os.path.join(path, "gen_regularisation_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the supervised phase
        # -------------------------------
        log.info("Building supervised phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.z_discriminator.trainable = False
        self.s_discriminator.trainable = False
        # self.infomax_net.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_sup")

        s_pred, _ = self.encoder(x)

        self.supervised_phase = Model(
            inputs=x,
            outputs=s_pred,
            name="supervised_phase"
        )

        plot_model(self.supervised_phase, os.path.join(path, "supervised_phase.png"), show_shapes=True)

        log.info("Building infomax phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = False
        self.s_discriminator.trainable = False
        # self.infomax_net.trainable = True

        z_info = Input(shape=(self.z_length,), name="z_info")
        s_info = Input(shape=(self.s_length,), name="s_info")

        y_drums_info, y_bass_info, y_guitar_info, y_strings_info = self.decoder([s_info, z_info])

        y = Concatenate(axis=-1, name="concat")([y_drums_info, y_bass_info, y_guitar_info, y_strings_info])

        s_info_pred, _ = self.encoder(y)

        # s_info_pred = self.infomax_net([Y_drums_info, Y_bass_info, Y_guitar_info, Y_strings_info])

        self.infomax_phase = Model(
            inputs=[s_info, z_info],
            outputs=s_info_pred,
            name="infomax_phase"
        )

        plot_model(self.infomax_phase, os.path.join(path, "infomax_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        log.info("Building adversarial autoencoder's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = True
        self.s_discriminator.trainable = True

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
        z_real = Input(shape=(self.z_length,), name="z")
        s_real = Input(shape=(self.s_length,), name="s")

        y_drums, y_bass, y_guitar, y_strings = self.reconstruction_phase(x)
        z_valid_real, z_valid_fake, z_valid_int, z_int = self.z_regularisation_phase([z_real, x])
        s_valid_real, s_valid_fake, s_valid_int, s_int = self.s_regularisation_phase([s_real, x])
        s_valid_gen, z_valid_gen = self.gen_regularisation_phase(x)
        s_pred = self.supervised_phase(x)
        s_infomax = self.infomax_phase([s_real, z_real])

        self.adversarial_autoencoder = Model(
            inputs=[s_real, z_real, x],
            outputs=[
                y_drums, y_bass, y_guitar, y_strings,
                s_valid_real, s_valid_fake, s_valid_int,
                z_valid_real, z_valid_fake, z_valid_int,
                s_valid_gen, z_valid_gen,
                s_pred,
                s_infomax
            ],
            name="adversarial_autoencoder"
        )

        # prepare gp losses
        self.s_gp_loss = partial(h.gradient_penalty_loss, averaged_samples=s_int)
        self.s_gp_loss.__name__ = "gradient_penalty_s"

        self.z_gp_loss = partial(h.gradient_penalty_loss, averaged_samples=z_int)
        self.z_gp_loss.__name__ = "gradient_penalty_z"

        self.adversarial_autoencoder.compile(
            loss=[
                "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy",
                "categorical_crossentropy",
                h.wasserstein_loss, h.wasserstein_loss, self.s_gp_loss,
                h.wasserstein_loss, h.wasserstein_loss, self.z_gp_loss,
                h.wasserstein_loss, h.wasserstein_loss,
                "binary_crossentropy",
                "binary_crossentropy"
            ],
            loss_weights=[
                self.reconstruction_weight, self.reconstruction_weight, self.reconstruction_weight,
                self.reconstruction_weight,
                self.regularisation_weight, self.regularisation_weight, self.regularisation_weight * self.s_lambda,
                self.regularisation_weight, self.regularisation_weight, self.regularisation_weight * self.z_lambda,
                self.regularisation_weight, self.regularisation_weight,
                self.supervised_weight,
                self.infomax_weight
            ],
            optimizer=self.aae_optim,
            metrics=[
                "categorical_accuracy",
                "binary_accuracy",
                h.output
            ]
        )
        plot_model(self.adversarial_autoencoder, os.path.join(path, "adversarial_autoencoder.png"), show_shapes=True)

    def train_v2(self, training_batches: List[list], test_batches: List[list]):
        epsilon_std = cfg.model_params["encoder_params"]["epsilon_std"]
        # create checkpoint and plots folder
        paths = {
            "interpolations": os.path.join(cfg.general_params["interpolations_path"], self.name),
            "autoencoded": os.path.join(cfg.general_params["autoencoded_path"], self.name),
            "checkpoints": os.path.join(cfg.general_params["checkpoints_path"], self.name),
            "plots": os.path.join(cfg.plots_path, self.name),
            "sampled": os.path.join(cfg.general_params["sampled_path"], self.name),
            "style_transfers": os.path.join(cfg.general_params["style_transfers_path"], self.name),
            "latent_sweeps": os.path.join(cfg.general_params["latent_sweeps_path"], self.name)
        }
        for key in paths:
            h.create_dirs(paths[key])

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
            "s_score_real": [],
            "s_score_fake": [],
            "s_gradient_penalty": [],
            "z_score_real": [],
            "z_score_fake": [],
            "z_gradient_penalty": [],
            "supervised_loss": [],
            "supervised_accuracy": [],
            "infomax_loss": [],
            "infomax_accuracy": []
        }

        vl_log = {
            "epoch": [],
            # "AE_loss_drums": [],
            # "AE_loss_bass": [],
            # "AE_loss_guitar": [],
            # "AE_loss_strings": [],
            "VL_AE_accuracy_drums": [],
            "VL_AE_accuracy_bass": [],
            "VL_AE_accuracy_guitar": [],
            "VL_AE_accuracy_strings": [],
            "VL_AE_accuracy_tot": [],
            # "s_score_real": [],
            # "s_score_fake": [],
            # "s_gradient_penalty": [],
            # "z_score_real": [],
            # "z_score_fake": [],
            # "z_gradient_penalty": [],
            # "supervised_loss": [],
            # "supervised_accuracy": []
            "VL_infomax_loss": [],
            "VL_infomax_accuracy": []
        }

        # ... let the training begin!
        pbc = 0
        pbc_tr = 0
        pbc_vl = 0
        annealing_first_stage = False
        annealing_second_stage = False
        annealing_third_stage = False

        len_training_set = sum(len(batch) for batch in training_batches)
        len_test_set = sum(len(batch) for batch in test_batches)
        # bar.update(0)
        for epoch in range(self.n_epochs):
            log.info("- Epoch", epoch + 1, "of", self.n_epochs)
            log.info("-- Number of TR batches:", len(training_batches))
            log.info("-- Number of VL batches:", len(test_batches))

            tr_queue = Queue(maxsize=128)

            def async_batch_generator_tr():
                for batch in training_batches:
                    tr_queue.put(batch)

            training_batch_thread = threading.Thread(target=async_batch_generator_tr)
            training_batch_thread.start()

            print("Training on training set...")
            # train on the training set
            for _ in range(len_training_set):
                x, y = tr_queue.get()

                n_chunks = x.shape[0]

                # Adversarial ground truth (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z from N(0,epsilon_std)
                z_real = np.random.normal(0, epsilon_std, (n_chunks, self.z_length))

                # draw s from B(s_length)
                # s is a k-hot vector of tags
                s_real = np.random.binomial(1, 0.5, size=(n_chunks, self.s_length))

                # Y_split = [ Y[:, :, : , t] for t in range(self.n_tracks) ]
                y_drums = y[:, :, :, 0]
                y_bass = y[:, :, :, 1]
                y_guitar = y[:, :, :, 2]
                y_strings = y[:, :, :, 3]

                aae_loss = self.adversarial_autoencoder.train_on_batch(
                    [s_real, z_real, x],
                    [
                        y_drums, y_bass, y_guitar, y_strings,
                        real_gt, fake_gt, dummy_gt,
                        real_gt, fake_gt, dummy_gt,
                        real_gt, real_gt,
                        s_real
                    ]
                )

                tr_log["AE_loss_drums"].append(aae_loss[1])
                tr_log["AE_loss_bass"].append(aae_loss[2])
                tr_log["AE_loss_guitar"].append(aae_loss[3])
                tr_log["AE_loss_strings"].append(aae_loss[4])
                tr_log["AE_loss_tot"].append(np.array([aae_loss[1], aae_loss[2], aae_loss[3], aae_loss[4]]).mean())

                tr_log["AE_accuracy_drums"].append(aae_loss[15])
                tr_log["AE_accuracy_bass"].append(aae_loss[18])
                tr_log["AE_accuracy_guitar"].append(aae_loss[21])
                tr_log["AE_accuracy_strings"].append(aae_loss[24])
                tr_log["AE_accuracy_tot"].append(
                    np.array([aae_loss[15], aae_loss[18], aae_loss[21], aae_loss[24]]).mean())

                tr_log["s_score_real"].append(aae_loss[29])
                tr_log["s_score_fake"].append(aae_loss[32])
                tr_log["s_gradient_penalty"].append(aae_loss[7])

                tr_log["z_score_real"].append(aae_loss[38])
                tr_log["z_score_fake"].append(aae_loss[41])
                tr_log["z_gradient_penalty"].append(aae_loss[10])

                tr_log["supervised_loss"].append(aae_loss[47])
                tr_log["supervised_accuracy"].append(aae_loss[50])

                tr_log["infomax_loss"].append(aae_loss[14])
                tr_log["infomax_accuracy"].append(aae_loss[55])

                if pbc_tr % 500 == 0:
                    log.info("Plotting stats...")
                    log.info("Regularisation weight:", k.get_value(self.regularisation_weight))
                    h.plot(paths["plots"], tr_log)

                if pbc_tr % 5000 == 0:
                    log.info("\nSaving checkpoint...")
                    self.save_checkpoint(paths["checkpoints"], pbc_tr)

                # annealing the regularisation part
                if pbc_tr > 1000 and not annealing_first_stage:
                    k.set_value(self.regularisation_weight, 0.0)
                    log.info("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_first_stage = True
                elif pbc_tr > 10000 and not annealing_second_stage:
                    k.set_value(self.regularisation_weight, 0.1)
                    log.info("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_second_stage = True
                elif pbc_tr > 15000 and not annealing_third_stage:
                    k.set_value(self.regularisation_weight, 0.2)
                    log.info("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_third_stage = True

                pbc += 1
                pbc_tr += 1

            # at the end of each epoch, we evaluate on the validation set
            vl_queue = Queue(maxsize=128)

            def async_batch_generator_vl():
                for batch in test_batches:
                    vl_queue.put(batch)

            validation_batch_thread = threading.Thread(target=async_batch_generator_vl)
            validation_batch_thread.start()

            log.info("Evaluating on validation set...")
            # evaluating on validation set

            vl_log_tmp = {
                "VL_AE_accuracy_drums": [],
                "VL_AE_accuracy_bass": [],
                "VL_AE_accuracy_guitar": [],
                "VL_AE_accuracy_strings": [],
                "VL_infomax_loss": [],
                "VL_infomax_accuracy": []
            }

            for _ in range(len_test_set):
                x, y = vl_queue.get()

                n_chunks = x.shape[0]

                # Adversarial ground truths (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z from N(0,epsilon_std)
                z_real = np.random.normal(0, epsilon_std, (n_chunks, self.z_length))

                # draw s from B(s_length)
                # s is a k-hot vector of tags
                s_real = np.random.binomial(1, 0.5, size=(n_chunks, self.s_length))

                # Y_split = [ Y[:, :, : , t] for t in range(self.n_tracks) ]
                y_drums = y[:, :, :, 0]
                y_bass = y[:, :, :, 1]
                y_guitar = y[:, :, :, 2]
                y_strings = y[:, :, :, 3]
                aae_loss = self.adversarial_autoencoder.test_on_batch(
                    [s_real, z_real, x],
                    [
                        y_drums, y_bass, y_guitar, y_strings,
                        real_gt, fake_gt, dummy_gt,
                        real_gt, fake_gt, dummy_gt,
                        real_gt, real_gt,
                        s_real
                    ]
                )
                vl_log_tmp["VL_AE_accuracy_drums"].append(aae_loss[15])
                vl_log_tmp["VL_AE_accuracy_bass"].append(aae_loss[18])
                vl_log_tmp["VL_AE_accuracy_guitar"].append(aae_loss[21])
                vl_log_tmp["VL_AE_accuracy_strings"].append(aae_loss[24])

                vl_log_tmp["VL_infomax_loss"].append(aae_loss[14])
                vl_log_tmp["VL_infomax_accuracy"].append(aae_loss[55])

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

            vl_log["VL_infomax_loss"].append(np.array(vl_log_tmp["VL_infomax_loss"]).mean())
            vl_log["VL_infomax_accuracy"].append(np.array(vl_log_tmp["VL_infomax_accuracy"]).mean())

            with open(os.path.join(paths["plots"], "log.json"), 'w') as f:
                json.dump(str(vl_log), f)

            h.plot(paths["plots"], vl_log)

    def save_checkpoint(self, path, epoch):
        self.encoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_encoder.h5"))
        self.decoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_decoder.h5"))
