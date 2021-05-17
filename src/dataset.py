import logging as log
import multiprocessing as mp
import os
from glob import glob
from typing import List, Optional

import numpy as np
import pretty_midi
import pypianoroll as pproll
from keras.utils import to_categorical

import config as cfg
import helper as h
import midi_cfg

log.getLogger(__name__)


def create_samples(pool: mp.Pool, midis: List[str]) -> List[str]:
    wrong_track_count = 0
    wrong_signature_count = 0

    cs = h.get_chunksize(midis)
    log.info(f"Creating samples out of {len(midis)} songs and saving them to {cfg.Paths.samples}...")
    log.info(" - Chunksize: {}".format(cs))
    log.info(" - Number of chunks: {}".format(float(len(midis) / cs)))
    for res in pool.imap_unordered(iterable=midis, func=_create_sample, chunksize=cs):
        if res is cfg.Exceptions.WRONG_TRACK_COUNT:
            wrong_track_count += 1
        if res is cfg.Exceptions.WRONG_SIGNATURE:
            wrong_signature_count += 1
    log.info(f" - {wrong_track_count} songs have a wrong track count.")
    log.info(f" - {wrong_signature_count} songs have a wrong signature and/or at least one signature change")
    samples = glob(cfg.Paths.samples + "/*sample*.npz")
    log.info(f" - {len(samples)} samples created.")
    return samples


def _create_sample(song: str) -> Optional[cfg.Exceptions]:
    name = os.path.split(song)[-1].split(".mid")[0]
    # name = song.split(os.sep)[-1].split(".mid")[0]
    name = name.split("(")[0]
    pm = pretty_midi.PrettyMIDI(song)
    for sig_change in pm.time_signature_changes:
        if not (sig_change.numerator == 4 and sig_change.denominator == 4):
            return cfg.Exceptions.WRONG_SIGNATURE
    song = pproll.from_pretty_midi(midi=pm, resolution=midi_cfg.resolution)
    tracks = get_instruments(song)

    if not tracks:
        return cfg.Exceptions.WRONG_TRACK_COUNT

    # combinations = list(itertools.product(guitar, bass, drums, string))
    combinations = tracks
    # Generate all combinations of our given tracks
    sample_count = 0
    for combination in combinations:
        # mt = pproll.Multitrack(tracks=list(combination), resolution=midi_cfg.resolution)
        mt = pproll.Multitrack(tracks=[combination], resolution=midi_cfg.resolution)

        mt = mt.binarize()
        mt.set_nonzeros(1)

        pianoroll = mt.stack()

        for j in range(0, pianoroll.shape[1] + 1, midi_cfg.phrase_size):
            window = pianoroll[:, j:j + midi_cfg.phrase_size, :]

            if window.shape[1] != midi_cfg.phrase_size:
                continue

            # keep only the phrases that have at most one bar of consecutive silence
            # for each track
            # bar_of_silences = np.array([0] * midi_cfg.n_tracks)
            # for track in range(midi_cfg.n_tracks):
            #     for k in range(0, window.shape[1] + 1, midi_cfg.bar_size):
            #         if window[track, k:k + midi_cfg.bar_size, :].sum() == 0:
            #             bar_of_silences[track] += 1

            # if the phrase is good, let's store it
            # if not any(bar_of_silences > 1):
            # data augmentation, random transpose bar
            shift = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
            tracks = []
            for track in range(midi_cfg.n_tracks):
                tracks.append(pproll.StandardTrack(pianoroll=window[track, :, :], is_drum=mt.tracks[track].is_drum))
            tmp = pproll.Multitrack(tracks=tracks)

            tmp.transpose(shift)
            path = os.path.join(cfg.Paths.samples, f"{name}_sample_{sample_count}.npz")
            np.savez_compressed(path, sample=tmp.stack())
            del tmp
            sample_count += 1


def create_batches(pool: mp.Pool, samples: List[str], batch_size=cfg.batch_size) -> List[str]:
    batches = list(get_chunks(samples, batch_size))

    batches = [(batch, i) for i, batch in enumerate(batches)]

    log.info("Creating batches out of {} samples...".format(len(samples)))
    log.info(" - Batch size: {}".format(batch_size))
    log.info(" - Number of batches: {}".format(len(batches)))
    pool.starmap(iterable=batches, func=_process_batch, chunksize=h.get_chunksize(batches))
    return glob(cfg.Paths.musae + "/**/batch[0-9]*.npz", recursive=True)


def _process_batch(batch: List[str], name):
    dest = []
    for sample in batch:
        with np.load(sample) as npz:
            dest.append(npz['sample'])
    dest = np.array(dest)
    preprocess_batch(dest, name)


def preprocess_single(sample) -> [np.ndarray]:
    x = sample.astype("float64")
    x = np.pad(x, ((0, 0), (0, 0), (0, 2)))

    # Add silent note
    for track in range(x.shape[0]):
        for ts in range(0, x.shape[1]):
            if all(x[track, ts, :-2] == 0):
                x[track, ts, -2] = 1

    # converting to categorical (keep only one note played at a time)
    tracks = []
    for t in range(midi_cfg.n_tracks):
        x_t = x[t, :, :]
        x_t = to_categorical(x_t.argmax(1), num_classes=midi_cfg.n_cropped_notes)
        x_t = np.expand_dims(x_t, axis=0)

        tracks.append(x_t)

    x = np.concatenate(tracks, axis=0)

    # Add held note
    for track in range(x.shape[0]):
        for ts in range(1, x.shape[1]):
            if np.array_equal(x[track, ts, :-1], x[track, ts - 1, :-1]):
                x[track, ts, -1] = 1

    # Zero the pianoroll, where there is a held note
    for track in range(x.shape[0]):
        for ts in range(1, x.shape[1]):
            if x[track, ts, -1] == 1:
                x[track, ts, :-1] = 0
    x = x.reshape((1, 1, midi_cfg.phrase_size, 130))

    return x


def preprocess_batch(dest, name: int) -> [np.ndarray, np.ndarray]:
    x = dest.astype("float64")
    x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, 2)))

    # Add silent note
    for sample in range(x.shape[0]):
        for track in range(x.shape[1]):
            for ts in range(0, x.shape[2]):
                if all(x[sample, track, ts, :-2] == 0):
                    x[sample, track, ts, -2] = 1

    # converting to categorical (keep only one note played at a time)
    tracks = []
    for t in range(midi_cfg.n_tracks):
        x_t = x[:, t, :, :]
        x_t = to_categorical(x_t.argmax(2), num_classes=midi_cfg.n_cropped_notes)
        x_t = np.expand_dims(x_t, axis=1)

        tracks.append(x_t)

    x = np.concatenate(tracks, axis=1)

    # Add held note
    for sample in range(x.shape[0]):
        for track in range(x.shape[1]):
            for ts in range(1, x.shape[2]):
                if np.array_equal(x[sample, track, ts, :-1], x[sample, track, ts - 1, :-1]):
                    x[sample, track, ts, -1] = 1

    # Zero the pianoroll, where there is a held note
    for sample in range(x.shape[0]):
        for track in range(x.shape[1]):
            for ts in range(1, x.shape[2]):
                if x[sample, track, ts, -1] == 1:
                    x[sample, track, ts, :-1] = 0
    # finally, use [0, 1] interval for ground truth Y and [-1, 1] interval for input/teacher forcing X
    y = x.copy()
    x[np.equal(x, 0)] = -1

    np.savez_compressed(os.path.join(cfg.Paths.batches, f"batch{name}"), x=x, y=y)

    del x, y


def get_chunks(samples: List[str], n: int) -> List[List[str]]:
    for i in range(0, len(samples), n):
        yield samples[i:i + n]


def get_instruments(song: pproll.Multitrack) -> List[pproll.Track]:
    tracks = []

    for track in song.tracks:
        p = track.program
        n = track.name
        n = n.lower()
        if ("violino" or "violen") in n:
            track.program = 40
            tracks.append(track)
        elif ("viola" or "altvioln") in n:
            track.program = 41
            tracks.append(track)
        elif ("violoncello" or "cello") in n:
            track.program = 42
            tracks.append(track)
        elif "pan flute" in n:
            track.program = 75
            tracks.append(track)
        elif "flute" in n:
            track.program = 73
            tracks.append(track)
        else:
            if 24 <= p <= 31:
                tracks.append(track)
            if 40 <= p <= 47:
                tracks.append(track)
            if 56 <= p <= 79:
                tracks.append(track)
    return tracks


def postprocess(predicted_tracks):
    samples = []
    for sample in range(predicted_tracks.shape[0]):
        tracks = []
        for track in range(midi_cfg.n_tracks):
            track = predicted_tracks[sample, track, :, :]
            track = to_categorical(track.argmax(1), num_classes=midi_cfg.n_cropped_notes)
            track = np.expand_dims(track, axis=0)
            tracks.append(track)
        x = np.concatenate(tracks, axis=0)
        samples.append(x)
    samples = np.array(samples)

    # copying previous timestep if held note is on
    for sample in range(predicted_tracks.shape[0]):
        for track in range(samples.shape[1]):
            for ts in range(1, samples.shape[2]):
                if samples[sample, track, ts, -2] == 1:
                    samples[sample, track, ts, :-2] = 0
                elif samples[sample, track, ts, -1] == 1:  # if held note is on
                    samples[sample, track, ts, :] = samples[sample, track, ts - 1, :]

    # drop silent note and held note (last two notes)
    # this leaves all 0s when there is a silent note (good)
    return samples[:, :, :, :-2]
