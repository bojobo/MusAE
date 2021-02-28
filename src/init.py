import logging as log
import os
import pickle
import zipfile as zf
from datetime import datetime as dt
from os import path as p

from sklearn.model_selection import train_test_split

import config as cfg
from dataset import MidiDataset
from model import MusAE_GM

# Initialize a global logger
log.basicConfig(
    filename=os.path.join(cfg.log_path, dt.now().strftime("%d-%m-%Y_%H.%M.%S") + ".txt"),
    format='%(levelname)s:  %(name)s:   %(asctime)s:    %(message)s',
    filemode='w',
    level=log.INFO)

if __name__ == '__main__':
    log.info("Using GPU: {}".format(tf.test.is_gpu_available()))
    log.info("Initializing...")
    # # Check if our training and test samples have already been created.
    # # If not, create them
    if (not os.path.exists(cfg.training_batches)) | (not os.path.exists(cfg.test_batches)):
        log.info("Extracting " + cfg.zip_path + "...")
        with zf.ZipFile(cfg.zip_path, 'r') as z:
            z.extractall(cfg.unzip_path)
        midi_count = 0
        for path, _, files in os.walk(cfg.unzip_path):
            for file in files:
                if not file.endswith(".mid"):
                    os.remove(p.join(path, file))
                else:
                    midi_count += 1
        log.info("Extracted " + str(midi_count) + " .midi files")
        d = MidiDataset()
        songs = [p.join(path, file) for path, _, files in os.walk(cfg.unzip_path) for file in files]
        training_songs, test_songs = train_test_split(songs, test_size=0.2)

        log.info("Creating training samples...")
        samples = d.create_samples(training_songs)
        log.info("Creating training batches...")
        pickle.dump(d.create_batches(samples), open(cfg.training_batches, 'wb'))

        del samples

        log.info("Creating test samples...")
        samples = d.create_samples(test_songs)
        log.info("Creating test batches...")
        pickle.dump(d.create_batches(samples), open(cfg.test_batches, 'wb'))
        del samples

    log.info("Loading training batches...")
    training_batches = pickle.load(open(cfg.training_batches, 'rb'))
    log.info("Loading test batches...")
    test_batches = pickle.load(open(cfg.test_batches, 'rb'))

    log.info("Initialising MusÆ...")
    aae = MusAE_GM()

    log.info("Traingin MusÆ...")
    aae.train(training_batches, test_batches)
