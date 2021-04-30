import itertools
import logging as log
import multiprocessing as mp
import os
from glob import glob
from typing import List, Union

import numpy as np
import pypianoroll as pproll
from keras.utils import to_categorical

import config as cfg
import helper as h
import midi_cfg

log.getLogger(__name__)


def _create_pianoroll(song: str) -> Union[cfg.Exceptions, int]:
    name = song.split("\\")[-1].split(".mid")[0]
    name = name.split("(")[0]
    base_song = pproll.read(path=song, resolution=midi_cfg.resolution)
    base_song.name = name
    # piano, others = get_instruments(base_song)
    guitar, bass, drums, strings = get_instruments(base_song)

    if (not guitar) or (not bass) or (not drums) or (not strings):
        return cfg.Exceptions.WRONG_TRACK_COUNT

    # if not piano:
    #     return cfg.Exceptions.WRONG_TRACK_COUNT
    # if not others:
    #     others = piano.copy()
    if len(strings) > 1:
        mt = base_song.copy()
        mt.tracks = strings
        blended = mt.blend(mode="max")
        track = pproll.Track(name="Others", program=48, is_drum=False, pianoroll=blended)
        tracks = []
        tracks.extend(guitar)
        tracks.extend(bass)
        tracks.extend(drums)
        tracks.append(track)
        base_song.tracks = tracks
    return _create_sample(base_song, name)


def create_samples(midis: List[str]) -> None:
    wrong_track_count = 0

    cs = h.get_chunksize(midis)
    with mp.Pool(processes=cfg.processes) as pool:
        log.info("Creating pianorolls and samples out of {} songs and saving them to {}...".format(len(midis),
                                                                                                   cfg.Paths.musae))
        log.info(" - Chunksize: {}".format(cs))
        log.info(" - Number of chunks: {}".format(float(len(midis) / cs)))
        pianoroll_count = 0
        sample_count = 0
        for res in pool.imap_unordered(iterable=midis, func=_create_pianoroll, chunksize=cs):
            if res is cfg.Exceptions.WRONG_TRACK_COUNT:
                wrong_track_count += 1
            if res is int:
                pianoroll_count += 1
                sample_count += res
        pool.close()
        pool.join()
    log.info(" - {} songs have a wrong track count.".format(wrong_track_count))
    log.info(" - {} pianorolls created.".format(pianoroll_count))
    log.info(" - {} samples created.".format(sample_count))


def _create_sample(song: pproll.Multitrack, prefix: str) -> int:
    guitar, bass, drums, strings = get_instruments(song)

    combinations = list(itertools.product(guitar, bass, drums, strings))

    # Generate all combinations of our given tracks
    sample_count = 0
    for combination in combinations:
        mt = pproll.Multitrack(tracks=list(combination))

        mt = mt.binarize()
        mt.set_nonzeros(1)

        pianoroll = mt.stack()

        for j in range(0, pianoroll.shape[1] + 1, midi_cfg.phrase_size):
            window = pianoroll[:, j:j + midi_cfg.phrase_size, :]

            if window.shape[1] != midi_cfg.phrase_size:
                continue

            # keep only the phrases that have at most one bar of consecutive silence
            # for each track
            bar_of_silences = np.array([0] * midi_cfg.n_tracks)
            for track in range(midi_cfg.n_tracks):
                for k in range(0, window.shape[1] + 1, midi_cfg.bar_size):
                    if window[track, k:k + midi_cfg.bar_size, :].sum() == 0:
                        bar_of_silences[track] += 1

            # if the phrase is good, let's store it
            if not any(bar_of_silences > 1):
                # data augmentation, random transpose bar
                shift = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
                tracks = []
                for track in range(midi_cfg.n_tracks):
                    tracks.append(pproll.Track(
                        pianoroll=window[track, :, :],
                        program=mt.tracks[track].program,
                        name=mt.tracks[track].name,
                        is_drum=mt.tracks[track].is_drum)
                    )
                tmp = pproll.Multitrack(tracks=tracks)

                tmp.transpose(shift)
                tmp = tmp.stack().reshape((midi_cfg.phrase_size, 128, midi_cfg.n_tracks))
                path = os.path.join(cfg.Paths.musae, "{}sample{}.npz".format(prefix, sample_count))
                np.savez_compressed(path, tmp)
                del tmp
                sample_count += 1
    return sample_count


def create_batches(batch_size=cfg.batch_size) -> None:
    samples = glob(cfg.Paths.musae + "/**/*sample[0-9]*.npz", recursive=True)
    batches = list(get_chunks(samples, batch_size))

    batches = [(batch, i) for i, batch in enumerate(batches)]

    with mp.Pool(processes=cfg.processes) as pool:
        log.info("Creating batches out of {} samples...".format(len(samples)))
        log.info(" - Batch size: {}".format(batch_size))
        log.info(" - Number of batches: {}".format(len(batches)))
        batch_count = 0
        for count in pool.starmap(iterable=batches, func=_process_batch, chunksize=h.get_chunksize(batches)):
            batch_count = batch_count + count
        pool.close()
        pool.join()
    # h.wait_on_disk(batch_count, cfg.Paths.musae, "./batch[0-9]*.npz")


def _process_batch(batch: List[str], name) -> int:
    dest = []
    for sample_name in batch:
        sample = np.load(os.path.join(cfg.Paths.musae, sample_name))
        dest.append(sample['arr_0'])
    dest = np.array(dest)
    preprocess(dest, name)
    return 1


def preprocess(dest: np.ndarray, name: int) -> [np.ndarray, np.ndarray]:
    x = np.pad(dest, ((0, 0), (0, 0), (0, 2), (0, 0)))

    for sample in range(x.shape[0]):
        for ts in range(0, x.shape[1]):
            for track in range(x.shape[3]):
                if all(x[sample, ts, :-2, track] == 0):
                    x[sample, ts, -2, track] = 1
                elif (ts > 0) and np.array_equal(x[sample, ts, :-1, track], x[sample, ts - 1, :-1, track]):
                    x[sample, ts, -1, track] = 1
                    x[sample, ts, :-1, track] = 0

    # converting to categorical (keep only one note played at a time)
    tracks = []
    for t in range(midi_cfg.n_tracks):
        x_t = x[:, :, :, t]
        x_t = to_categorical(x_t.argmax(2), num_classes=midi_cfg.n_cropped_notes)
        x_t = np.expand_dims(x_t, axis=-1)

        tracks.append(x_t)

    x = np.concatenate(tracks, axis=-1)

    # finally, use [0, 1] interval for ground truth Y and [-1, 1] interval for input/teacher forcing X
    y = x.copy()
    x[x == 0] = -1

    np.savez_compressed(os.path.join(cfg.Paths.musae, "batch{}".format(name)), x=x, y=y)

    del x, y


def get_chunks(samples: List[str], n: int) -> List[List[str]]:
    for i in range(0, len(samples), n):
        yield samples[i:i + n]


def get_instruments(song: pproll.Multitrack) -> [list, list, list, list]:
    # piano = []
    # other = []
    #
    # for i, track in enumerate(song.tracks):
    #     if track.program in range(8):
    #         track.name = "Piano"
    #         piano.append(track)
    #     else:
    #         track.name = "Others"
    #         other.append(track)
    #
    # return piano, other
    guitar_tracks = []
    bass_tracks = []
    drums_tracks = []
    string_tracks = []

    for i, track in enumerate(song.tracks):
        if track.is_drum:
            track.name = "Drums"
            drums_tracks.append(track)
        elif 0 <= track.program <= 31:
            track.name = "Guitar"
            guitar_tracks.append(track)
        elif 32 <= track.program <= 39:
            track.name = "Bass"
            bass_tracks.append(track)
        else:
            string_tracks.append(track)

    return guitar_tracks, bass_tracks, drums_tracks, string_tracks


def postprocess(predicted_tracks):
    tracks = []
    for track in predicted_tracks:
        track = track[0, :, :]
        track = to_categorical(track.argmax(1), num_classes=midi_cfg.n_cropped_notes)
        track = np.expand_dims(track, axis=-1)
        tracks.append(track)

    x = np.concatenate(tracks, axis=-1)

    # copying previous timestep if held note is on
    for ts in range(1, x.shape[0]):
        for track in range(x.shape[2]):
            if x[ts, -2, track] == 1:
                x[ts, :-2, track] = 0
            elif x[ts, -1, track] == 1:  # if held note is on
                x[ts, :, track] = x[ts - 1, :, track]

    # drop silent note and held note (last two notes)
    # this leaves all 0s when there is a silent note (good)
    x = x[:, :-2, :]
    return x
