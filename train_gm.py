from __future__ import print_function, division

import json
import os
import pprint
import random
import threading
from functools import partial
from queue import Queue

import numpy as np
from keras import backend as k
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import config as cfg
import decoders
import discriminators
import encoders
import helper as h

pp = pprint.PrettyPrinter(indent=4)


class MusAE_GM:
    def __init__(self):
        # setting params as class attributes
        self.plots_path = cfg.general_params["plots_path"]

        self.name = cfg.model_params["name"]
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

        # Initialize
        self.len_tr_set = -1
        self.len_vl_set = -1

        print("Initialising encoder...")
        self.encoder = encoders.build_encoder_z()

        print("Initialising decoder...")
        self.decoder = decoders.build_decoder_z_flat()

        print("Initialising z discriminator...")
        self.z_discriminator = discriminators.build_gaussian_mixture_discriminator()

        path = os.path.join(self.plots_path, self.name, "models")
        h.create_dirs(path)

        print("Saving model plots..")
        plot_model(self.encoder, os.path.join(path, "encoder.png"), show_shapes=True)
        plot_model(self.decoder, os.path.join(path, "decoder.png"), show_shapes=True)
        plot_model(self.z_discriminator, os.path.join(path, "z_discriminator.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the Adversarial Autoencoder
        # -------------------------------
        print("Building reconstruction phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_recon")
        z_recon = self.encoder(x)
        y_drums, y_bass, y_guitar, y_strings = self.decoder(z_recon)

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
        print("Building z regularisation phase's computational graph...")
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.z_discriminator.trainable = True

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_z_reg")
        y = Input(shape=(self.s_length,), name="y_reg")

        z_real = Input(shape=(self.z_length,), name="z_reg")
        z_fake = self.encoder(x)

        z_int = h.RandomWeightedAverage(name="weighted_avg_z")([z_real, z_fake])

        z_valid_real = self.z_discriminator([z_real, y])
        z_valid_fake = self.z_discriminator([z_fake, y])
        z_valid_int = self.z_discriminator([z_int, y])

        self.z_regularisation_phase = Model(
            [z_real, x, y],
            [z_valid_real, z_valid_fake, z_valid_int, z_int],
            name="z_regularisation_phase"
        )
        plot_model(self.z_regularisation_phase, os.path.join(path, "z_regularisation_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        print("Building generator regularisation phase's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.z_discriminator.trainable = False
        # self.infomax_net.trainable = False

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X_gen_reg")
        y = Input(shape=(self.s_length,), name="y_gen_reg")

        z_gen = self.encoder(x)
        z_valid_gen = self.z_discriminator([z_gen, y])

        self.gen_regularisation_phase = Model(
            inputs=[x, y],
            outputs=z_valid_gen,
            name="gen_regularisation_phase"
        )
        plot_model(self.gen_regularisation_phase, os.path.join(path, "gen_regularisation_phase.png"), show_shapes=True)

        # -------------------------------
        # Construct Computational Graph
        # for the generator (encoder)
        # -------------------------------
        print("Building adversarial autoencoder's computational graph...")
        self.encoder.trainable = True
        self.decoder.trainable = True
        self.z_discriminator.trainable = True

        x = Input(shape=(self.phrase_size, self.n_cropped_notes, self.n_tracks), name="X")
        y = Input(shape=(self.s_length,), name="y")
        z_real = Input(shape=(self.z_length,), name="z")

        y_drums, y_bass, y_guitar, y_strings = self.reconstruction_phase(x)
        z_valid_real, z_valid_fake, z_valid_int, z_int = self.z_regularisation_phase([z_real, x, y])
        z_valid_gen = self.gen_regularisation_phase([x, y])

        self.adversarial_autoencoder = Model(
            inputs=[z_real, x, y],
            outputs=[
                y_drums, y_bass, y_guitar, y_strings,
                z_valid_real, z_valid_fake, z_valid_int,
                z_valid_gen
            ],
            name="adversarial_autoencoder"
        )

        self.z_gp_loss = partial(h.gradient_penalty_loss, averaged_samples=z_int)
        self.z_gp_loss.__name__ = "gradient_penalty_z"

        self.adversarial_autoencoder.compile(
            loss=[
                "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy",
                "categorical_crossentropy",
                h.wasserstein_loss, h.wasserstein_loss, self.z_gp_loss,
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
        plot_model(self.adversarial_autoencoder, os.path.join(path, "adversarial_autoencoder.png"), show_shapes=True)

        # Init prior distribution
        print("Initialising prior distribution...")
        self.prior_mean = np.empty(shape=(self.s_length + 1, self.z_length))
        self.prior_cov = np.empty(shape=(self.s_length + 1, self.z_length, self.z_length))

        n = self.s_length + 1
        for i in range(self.s_length + 1):
            self.prior_mean[i] = [1 * np.cos((i * 2 * np.pi) / n), 1 * np.sin((i * 2 * np.pi) / n)] + [0 for _ in range(
                self.z_length - 2)]

            v1 = [np.cos((i * 2 * np.pi) / n), np.sin((i * 2 * np.pi) / n)]
            v2 = [-np.sin((i * 2 * np.pi) / n), np.cos((i * 2 * np.pi) / n)]

            a1 = .1  # radial axis (center-to-outer)
            a2 = .001  # tangent axis (along the circle)

            m = np.eye(self.z_length)
            s = np.eye(self.z_length)
            np.fill_diagonal(s, [a1] + [a2 for _ in range(self.z_length - 1)])
            m[0:2, 0:2] = np.vstack((v1, v2)).T

            tmp = np.dot(np.dot(m, s), np.linalg.inv(m))

            self.prior_cov[i] = tmp

    def train_v2(self, dataset):
        # create checkpoint and plots folder
        paths = {
            "checkpoints": os.path.join(cfg.general_params["checkpoints_path"], self.name),
            "plots": os.path.join(self.plots_path, self.name),
        }
        for key in paths:
            if not os.path.exists(paths[key]):
                os.makedirs(paths[key])

        print("Splitting dataset...")
        batches_path = os.path.join(cfg.general_params["dataset_path"], "batches", "X")
        _, _, files = next(os.walk(batches_path))
        tr_set, vl_set = train_test_split(files, test_size=cfg.training_params["test_size"])
        del files
        self.len_tr_set = len(tr_set)
        self.len_vl_set = len(vl_set)

        print("Number of TR batches:", self.len_tr_set)
        print("Number of VL batches:", self.len_vl_set)

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

        # ... let the training begin!
        pbc = 0
        pbc_tr = 0
        pbc_vl = 0
        annealing_first_stage = False
        annealing_second_stage = False
        annealing_third_stage = False
        # bar.update(0)
        for epoch in range(self.n_epochs):
            print("- Epoch", epoch + 1, "of", self.n_epochs)
            print("-- Number of TR batches:", self.len_tr_set)
            print("-- Number of VL batches:", self.len_vl_set)

            print("Generating training batches...")
            tr_queue = Queue(maxsize=128)

            def async_batch_generator_tr():
                tr_batches = list(range(self.len_tr_set))
                random.shuffle(tr_batches)
                for i in tr_batches:
                    tr_queue.put(dataset.select_batch(i), block=True)

            training_batch_thread = threading.Thread(target=async_batch_generator_tr)
            training_batch_thread.start()

            print("Training on training set...")
            # train on the training set
            for _ in range(self.len_tr_set):
                x, y, label = tr_queue.get(block=True)
                label = label[:, :self.s_length]

                n_chunks = x.shape[0]

                # Adversarial ground truth (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z prior from mixture of gaussian
                label = self.from_khot_to_1hot(label)
                z_real = self.sample_mixture_of_gaussian(self.s_length + 1, label)

                y_drums = y[:, :, :, 0]
                y_bass = y[:, :, :, 1]
                y_guitar = y[:, :, :, 2]
                y_strings = y[:, :, :, 3]

                aae_loss = self.adversarial_autoencoder.train_on_batch(
                    [z_real, x, label],
                    [
                        y_drums, y_bass, y_guitar, y_strings,
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
                    print("\nPlotting stats...")
                    # print("Regularisation weight:", K.get_value(self.regularisation_weight))
                    h.plot(paths["plots"], tr_log)
                    print("TR batches in the queue: ", tr_queue.qsize())

                if pbc_tr % 5000 == 0:
                    print("\nSaving checkpoint...")
                    self.save_checkpoint(paths["checkpoints"], pbc_tr)
                    print("TR batches in the queue: ", tr_queue.qsize())

                # annealing the regularisation weight
                if pbc_tr > 1000 and not annealing_first_stage:
                    k.set_value(self.regularisation_weight, 0.0)
                    print("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_first_stage = True
                elif pbc_tr > 5000 and not annealing_second_stage:
                    k.set_value(self.regularisation_weight, 0.1)
                    print("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_second_stage = True
                elif pbc_tr > 10000 and not annealing_third_stage:
                    k.set_value(self.regularisation_weight, 0.2)
                    print("Regularisation weight annealed to ", k.get_value(self.regularisation_weight))
                    annealing_third_stage = True

                pbc += 1
                pbc_tr += 1
            # at the end of each epoch, we evaluate on the validation set
            print("Generating validation batches...")
            vl_queue = Queue(maxsize=128)

            def async_batch_generator_vl():
                vl_batches = range(self.len_vl_set)
                # random.shuffle(vl_batches)
                for i in vl_batches:
                    vl_queue.put(dataset.select_batch(i), block=True)

            validation_batch_thread = threading.Thread(target=async_batch_generator_vl)
            validation_batch_thread.start()

            print("\nEvaluating on validation set...")
            # evaluating on validation set

            vl_log_tmp = {
                "VL_AE_accuracy_drums": [],
                "VL_AE_accuracy_bass": [],
                "VL_AE_accuracy_guitar": [],
                "VL_AE_accuracy_strings": [],
            }

            for _ in range(self.len_vl_set):
                x, y, label = vl_queue.get(block=True)
                label = label[:, :self.s_length]

                n_chunks = x.shape[0]

                # Adversarial ground truths (wasserstein)
                real_gt = -np.ones((n_chunks, 1))
                fake_gt = np.ones((n_chunks, 1))
                dummy_gt = np.zeros((n_chunks, 1))  # Dummy gt for gradient penalty (not actually used)

                # draw z prior from mixture of gaussian
                label = self.from_khot_to_1hot(label)
                z_real = self.sample_mixture_of_gaussian(self.s_length + 1, label)

                y_drums = y[:, :, :, 0]
                y_bass = y[:, :, :, 1]
                y_guitar = y[:, :, :, 2]
                y_strings = y[:, :, :, 3]
                aae_loss = self.adversarial_autoencoder.test_on_batch(
                    [z_real, x, label],
                    [
                        y_drums, y_bass, y_guitar, y_strings,
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

            print("Saving validation accuracy...")
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

    def from_khot_to_1hot(self, labels):
        result = []
        for label in labels:
            m = []
            for i in range(len(label)):
                if label[int(i)] == 1:
                    m.append(int(i))

            new_l = np.zeros(self.s_length)

            if m:
                choice = np.random.choice(m)
                new_l[choice] = 1

            result.append(new_l)

        return np.array(result)

    # sample from a "flower-shaped" mixture of gaussian (such as the AAE paper)
    def sample_mixture_of_gaussian(self, n, labels):
        vectors = []

        for index, y in enumerate(labels):
            if not np.any(y):
                i = n - 1
            else:
                i = np.argmax(y)

            vec = np.random.multivariate_normal(mean=self.prior_mean[i], cov=self.prior_cov[i], size=1)
            vectors.append(vec)

        return np.array(vectors).reshape(-1, self.z_length)  # , labels)

    def save_checkpoint(self, path, epoch):
        self.encoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_encoder.h5"))
        self.decoder.save_weights(os.path.join(path, str(epoch) + "_MusAE_decoder.h5"))
