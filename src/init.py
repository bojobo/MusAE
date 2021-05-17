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
from tensorboard.plugins import projector

import config as cfg
import dataset
import helper as h
import midi_cfg
import model_cfg
from decoders import DecoderZFlat
from encoders import EncoderZ
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
        pproll.from_pretty_midi(midi=PrettyMIDI(path), resolution=midi_cfg.resolution)
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


def _reconstruct():
    log.info("Choosing a random sample and autoencoding it...")
    samples = glob(cfg.Paths.samples + "/*sample*.npz")
    sample = np.random.choice(samples)
    sample = np.load(sample)
    sample = sample['sample']
    tracks = []
    for track in range(sample.shape[0]):
        t = pproll.Track(pianoroll=sample[track, :, :], program=0)
        t = t.binarize()
        tracks.append(t)

    song = pproll.Multitrack(tracks=tracks, resolution=midi_cfg.resolution)
    song = song.set_nonzeros(1)
    pproll.write(multitrack=song, path=os.path.join(cfg.Paths.generated, "original.mid"))

    sample = dataset.preprocess_single(sample)
    e = best_encoder.predict(sample)
    d = best_decoder.predict(e)
    d = d.reshape((1, 1, midi_cfg.phrase_size, 130))
    reconstructed = dataset.postprocess(d)
    tracks = []
    for sample in range(reconstructed.shape[0]):
        for track in range(reconstructed.shape[1]):
            t = pproll.Track(pianoroll=reconstructed[sample, track, :, :], program=0)
            t = t.binarize()
            tracks.append(t)

        song = pproll.Multitrack(tracks=tracks, resolution=midi_cfg.resolution)
        song = song.set_nonzeros(1)
        pproll.write(multitrack=song, path=os.path.join(cfg.Paths.generated, f"reconstructed{sample}.mid"))


def _visualize():
    log.info("Visualizing all samples with tensorboard/t-sne...")
    samples = glob(cfg.Paths.samples + "/*sample*.npz")
    metadata = os.path.join(cfg.Paths.latent_space, 'metadata.txt')
    with open(metadata, 'w') as m:
        vecs = []
        count = 0
        for sample in samples:
            sample_name = sample.split(os.sep)[-1]
            composer = sample_name.split("_")[0]
            m.write(f"{composer}\n")
            npz = np.load(sample)
            npz = npz['sample']
            npz = dataset.preprocess_single(npz)
            encoding = best_encoder.predict(npz)
            vecs.append(np.squeeze(encoding))
            count += 1
    c = projector.ProjectorConfig()
    vecs = np.asarray(vecs)
    vecs = tf.Variable(vecs, trainable=False, name='vectors')
    embed = c.embeddings.add()
    embed.tensor_name = vecs.name
    embed.metadata_path = metadata
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(cfg.Paths.latent_space, 'meta.ckpt'))
    writer = tf.summary.FileWriter(cfg.Paths.latent_space, sess.graph)
    projector.visualize_embeddings(writer, c)
    sess.close()


if __name__ == '__main__':
    log.info("Using GPU: {}".format(tf.test.is_gpu_available()))
    if cfg.preprocess:
        _prepocess()

    aae = MusAE_GM()
    aae.train()

    log.info("Loading best encoder...")
    best_encoder = k.models.load_model(cfg.Resources.best_encoder,
                                       custom_objects={"EncoderZ": EncoderZ, 'k': k.backend, "model_cfg": model_cfg})
    log.info("Loading best decoder...")
    best_decoder = k.models.load_model(cfg.Resources.best_decoder, custom_objects={"DecoderZFlat": DecoderZFlat})

    # log.info("Generating {} new samples...".format(10))
    # _generate_samples(best_decoder)
    _reconstruct()
    _visualize()

    log.info("Done!")
