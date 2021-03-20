import logging as log
import os
import shutil
from os import path as p

import tensorflow as tf

import config as cfg
import helper as h
from dataset import MidiDataset
from model import MusAE_GM

# Initialize a global logger
log.basicConfig(
    filename=cfg.Resources.log_txt,
    format='%(levelname)s:  %(name)s:   %(asctime)s:    %(message)s',
    filemode='w',
    level=log.INFO)

if __name__ == '__main__':
    log.info("Using GPU: {}".format(tf.test.is_gpu_available()))
    log.info("Initializing...")
    # # Check if our training and test samples have already been created.
    # # If not, create them
    if cfg.preprocess:
        log.info("Extracting {} to {}...".format(cfg.Resources.dataset_zip, cfg.Paths.unzip))
        shutil.unpack_archive(cfg.Resources.dataset_zip, cfg.Paths.unzip)
        songs = []
        for path, _, files in os.walk(cfg.Paths.unzip):
            for file in files:
                if not file.endswith(".mid"):
                    os.remove(p.join(path, file))
                else:
                    songs.append(p.join(path, file))
        log.info("Extracted {} MIDI files".format(len(songs)))
        d = MidiDataset(songs)
        del songs

        # analyzer.track_time_signature_changes()
        # analyzer.track_instruments()

        # log.info("Creating test batches...")
        # pickle.dump(d.create_batches(samples), open(cfg.test_batches, 'wb'))
        # del samples

    # log.info("Loading training batches...")
    # training_batches = pickle.load(open(cfg.training_batches, 'rb'))
    # log.info("Loading test batches...")
    # test_batches = pickle.load(open(cfg.test_batches, 'rb'))
    #
    log.info("Initialising MusÆ...")
    aae = MusAE_GM()

    log.info("Training MusÆ...")
    aae.train()

    h.cleanup()
