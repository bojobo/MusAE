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
    # with mp.Pool(processes=cfg.processes) as pool:
    #     log.info("Filtering out corrupt MIDI files (i.e. PrettyMidi cannot parse it and PyPianoroll"
    #              "cannot create a pianoroll out of it)...")
    #     pool.map(iterable=midis, func=_filter, chunksize=h.get_chunksize(midis))
    #     pool.close()
    #     pool.join()
    # log.info(" - Done!")
    dataset.create_samples(midis)
    dataset.create_batches()


def _filter(path: str) -> None:
    try:
        mt = pproll.from_pretty_midi(midi=PrettyMIDI(path), resolution=midi_cfg.resolution)
        mt.tempo = mt.tempo[:, 0]
        mt.downbeat = mt.downbeat[:, 0]
        mt.validate()
    except:
        os.remove(path)


def _generate_samples(decoder: k.Model):
    z = np.random.randn(1, 512)
    predicted = decoder.predict(z)
    x = dataset.postprocess(predicted)

    guitar = pproll.Track(pianoroll=x[:, :, 0], program=0)
    bass = pproll.Track(pianoroll=x[:, :, 1], program=32)
    drums = pproll.Track(pianoroll=x[:, :, 2], is_drum=True)
    strings = pproll.Track(pianoroll=x[:, :, 3], program=48)

    guitar = guitar.binarize()
    bass = bass.binarize()
    drums = drums.binarize()
    strings = strings.binarize()

    song = pproll.Multitrack(tracks=[guitar, bass, drums, strings], resolution=midi_cfg.resolution)
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
    # best_encoder = k.models.load_model(cfg.Resources.best_encoder, custom_objects={'k': k.backend})
    log.info("Loading best decoder...")
    best_decoder = k.models.load_model(cfg.Resources.best_decoder)

    # pianoroll = pproll.load(os.path.join(cfg.Paths.samples, "bach_art_of_fugue_contrapunctum_2_(nc)wittenburg_0.npz"))
    # pproll.write(os.path.join(cfg.Paths.out, "original.mid"), pianoroll)
    # arr = pianoroll.stack().reshape((midi_cfg.phrase_size, 128, midi_cfg.n_tracks))
    # dest = [arr]
    # dest = np.array(dest)
    # _, y = dataset.preprocess(dest)
    # z = best_encoder.predict(y)
    # piano, others = best_decoder.predict(z)
    # piano = piano[0, :, :]
    # others = others[0, :, :]
    # x = dataset.postprocess(piano, others)
    # # x = x.reshape((midi_cfg.phrase_size, 128, 2))
    # piano = pproll.Track(pianoroll=x[:, :, 0], program=0, is_drum=False, name="Piano")
    # others = pproll.Track(pianoroll=x[:, :, 1], program=48, is_drum=False, name="Others")
    # piano = piano.binarize()
    # others = others.binarize()
    # song = pproll.Multitrack(tracks=[piano, others], resolution=midi_cfg.beat_resolution)
    # song = song.set_nonzeros(1)
    # song.validate()
    # pproll.write(os.path.join(cfg.Paths.out, "reconstructed.mid"), song)
    log.info("Generating {} new samples...".format(10))
    _generate_samples(best_decoder)

    log.info("Done!")
