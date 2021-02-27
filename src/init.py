import logging as log
import os
import pickle
import zipfile as zf
from datetime import datetime as dt
from os import path as p

from sklearn.model_selection import train_test_split

import config as cfg
from dataset import MidiDataset
from train import MusAE

# Initialize a global logger
log.basicConfig(
    filename=os.path.join(cfg.log_path, dt.now().strftime("%d-%m-%Y_%H.%M.%S") + ".txt"),
    format='%(levelname)s:  %(name)s:   %(asctime)s:    %(message)s',
    filemode='w',
    level=log.INFO)

if __name__ == '__main__':
    log.info("Started initialization...")
    # Check if our training and test samples have already been created.
    # If not, create them
    if (not os.path.exists(cfg.training_samples)) | (not os.path.exists(cfg.test_samples)):
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
        pickle.dump(samples, open(cfg.training_samples, 'wb'))
        log.info("Creating training batches...")
        pickle.dump(d.create_batches(samples), open(cfg.training_batches, 'wb'))

        log.info("Creating test samples...")
        samples = d.create_samples(test_songs)
        pickle.dump(samples, open(cfg.test_samples, 'wb'))
        log.info("Creating test batches...")
        pickle.dump(d.create_batches(samples), open(cfg.test_batches, 'wb'))
        del samples

    log.info("Loading training batches...")
    training_batches = pickle.load(open(cfg.training_batches, 'rb'))
    log.info("Loading test batches...")
    test_batches = pickle.load(open(cfg.test_batches, 'rb'))

    log.info("Initialising MusÆ...")
    aae = MusAE()

    log.info("Traingin MusÆ...")
    aae.train_v2(training_batches, test_batches)
