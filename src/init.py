import logging as log
import multiprocessing as mp
import os
import shutil
from glob import glob

import keras as k
import numpy as np
import pypianoroll as pproll
import tensorflow as tf
from pretty_midi import PrettyMIDI

import config as cfg
import dataset
import helper as h
import midi_cfg
from model import MusAE_GM

# Initialize a global logger
log.basicConfig(
    filename=cfg.Resources.log_txt,
    format='%(levelname)s:  %(name)s:   %(asctime)s:    %(message)s',
    filemode='w',
    level=log.INFO)


def _prepocess():
    log.info("Extracting {} to {}...".format(cfg.Resources.dataset_zip, cfg.Paths.unzip))
    shutil.unpack_archive(cfg.Resources.dataset_zip, cfg.Paths.unzip)
    midis = glob(cfg.Paths.unzip + "/**/*.mid", recursive=True)
    log.info("Original .ZIP file contains {} .mid files.".format(len(midis)))
    with mp.Pool(processes=cfg.processes) as pool:
        log.info("Filtering out corrupt MIDI files (i.e. PrettyMidi cannot parse it and PyPianoroll"
                 "cannot create a pianoroll out of it)...")
        pool.map(iterable=midis, func=_filter, chunksize=h.get_chunksize(midis))
        log.info(" - Done")
        midis = glob(cfg.Paths.unzip + "/**/*.mid", recursive=True)
        samples = dataset.create_samples(pool=pool, midis=midis)
        dataset.create_batches(pool=pool, samples=samples)
        pool.close()
        pool.join()
    log.info(" - Done!")


def _filter(path: str) -> None:
    try:
        mt = pproll.from_pretty_midi(midi=PrettyMIDI(path), resolution=midi_cfg.resolution)
        mt.downbeat = mt.downbeat[:, 0]
        mt.tempo = mt.tempo[:, 0]
        mt.validate()
    except:
        os.remove(path)


def _generate_samples(decoder: k.Model):
    z = np.random.randn(1, 512)
    predicted = decoder.predict(z)
    x = dataset.postprocess(predicted)

    guitar = pproll.Track(pianoroll=predicted[0, :, :], program=0)

    guitar = guitar.binarize()

    song = pproll.Multitrack(tracks=[guitar], resolution=midi_cfg.resolution)
    song = song.set_nonzeros(1)
    song.validate()
    pproll.write(multitrack=song, path=os.path.join(cfg.Paths.generated, f"{0}.mid"))


if __name__ == '__main__':
    log.info("Using GPU: {}".format(tf.test.is_gpu_available()))
    if cfg.preprocess:
        _prepocess()

    aae = MusAE_GM()
    aae.train()

    # log.info("Loading best encoder...")
    # best_encoder = k.models.load_model(cfg.Resources.best_encoder, custom_objects={"EncoderZ": EncoderZ, 'k': k.backend, "model_cfg": model_cfg})
    # log.info("Loading best decoder...")
    # best_decoder = k.models.load_model(cfg.Resources.best_decoder, custom_objects={"DecoderZFlat": DecoderZFlat})

    # log.info("Generating {} new samples...".format(10))
    # _generate_samples(best_decoder)

    log.info("Done!")
