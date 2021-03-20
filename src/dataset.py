import itertools
import logging as log
import multiprocessing as mp
import os
from typing import List, Union, Optional

import numpy as np
import pretty_midi as pm
import pypianoroll as pproll
from keras.utils import to_categorical

import config as cfg
import helper as h
import midi_cfg

log.getLogger(__name__)


def _create_pianoroll(song: str) -> Union[cfg.Exceptions, str]:
    try:
        pm.PrettyMIDI(song)
    except:
        return cfg.Exceptions.EXCEPTION_PARSING_PRETTY_MIDI

    try:
        name = song.split("\\")[-1].split(".mid")[0]
        base_song = pproll.parse(song, beat_resolution=midi_cfg.beat_resolution, name=name)
        piano, others = get_instruments(base_song)

        if (not others) or (not piano):
            return cfg.Exceptions.WRONG_TRACK_COUNT
        base_song.merge_tracks(others, mode="max", program=48, name="Others", remove_merged=True)
    except:
        return cfg.Exceptions.EXCPETION_PARSING_PYPIANOROLL
    pproll.save(os.path.join(cfg.Paths.pianorolls, name), base_song)
    return "{}.{}".format(base_song.name, "npz")


def create_pianorolls(songs: list) -> List[str]:
    filtered = []  # Contains pypianoroll representations
    exceptions_parsing_pretty_midi = 0
    exceptions_parsing_pypianoroll = 0
    wrong_track_count = 0

    cs = h.get_chunksize(songs)
    with mp.Pool(processes=cfg.processes) as pool:
        log.info("Creating pianorolls out of {} songs...".format(len(songs)))
        log.info("Number of processes: {}".format(cfg.processes))
        log.info("Chunksize: {}".format(cs))
        log.info("Number of chunks: {}".format(float(len(songs) / cs)))
        for i, res in enumerate(pool.imap_unordered(iterable=songs, func=_create_pianoroll, chunksize=cs)):
            if (i + 1) % cs == 0:
                log.info("Processed {} chunks...".format(round(i / cs)))
            if res is cfg.Exceptions.EXCEPTION_PARSING_PRETTY_MIDI:
                exceptions_parsing_pretty_midi += 1
            elif res is cfg.Exceptions.EXCPETION_PARSING_PYPIANOROLL:
                exceptions_parsing_pypianoroll += 1
            elif res is cfg.Exceptions.WRONG_TRACK_COUNT:
                wrong_track_count += 1
            else:
                filtered.append(res)

    log.info("{} songs have thrown an exception while parsing PrettyMidi.".format(exceptions_parsing_pretty_midi))
    log.info("{} songs have thrown an exception while parsing PyPianoRoll.".format(exceptions_parsing_pypianoroll))
    log.info("{} songs have a wrong track count.".format(wrong_track_count))
    log.info("Therefore, {} pianorolls have been created".format(len(filtered)))
    return filtered


def get_chunks(samples: List[str], n: int) -> List[List[str]]:
    for i in range(0, len(samples), n):
        yield samples[i:i + n]


def get_instruments(song: pproll.Multitrack) -> [list, list]:
    piano = []
    other = []

    for i, track in enumerate(song.tracks):
        if track.program in range(8):
            track.name = "Piano"
            piano.append(i)
        else:
            track.name = "Others"
            other.append(i)

    return piano, other


class MidiDataset:
    def __init__(self, songs: List[str]):
        pianorolls = create_pianorolls(songs)
        samples = self.create_samples(pianorolls)
        self.create_batches(samples)

    # warning: tends to use a lot of storage (disk) space
    def create_batches(self, samples, batch_size=cfg.batch_size):
        batches = list(get_chunks(samples, batch_size))

        log.info("Number of samples to create batches of: {}".format(len(samples)))
        log.info("Batch size: {}".format(batch_size))
        log.info("Number of batches: {}".format(len(batches)))

        cs = h.get_chunksize(batches)
        with mp.Pool(processes=cfg.processes) as pool:
            for i, res in enumerate(pool.imap_unordered(iterable=batches, func=self._process_batch, chunksize=cs)):
                if (i + 1) % cs == 0:
                    log.info("Processed {} batches...".format(round(i + 1)))
                np.save(os.path.join(cfg.Paths.x, str(i)), res[0])
                np.save(os.path.join(cfg.Paths.y, str(i)), res[1])
                # batches_x.append(res[0])
                # batches_y.append(res[1])

    def _process_batch(self, batch: List[str]) -> [np.ndarray, np.ndarray]:
        dest = []
        for sample_name in batch:
            sample_name = "{}.{}".format(sample_name, "npz")
            sample = pproll.load(os.path.join(cfg.Paths.samples, sample_name))
            dest.append(sample.get_stacked_pianoroll())
        dest = np.array(dest)
        x, y = self.preprocess(dest)
        return x, y

    def preprocess(self, x: np.ndarray) -> [np.ndarray, np.ndarray]:
        # if silent timestep (all 0), then set silent note to 1, else set
        # silent note to 0
        def pad_with(vector, pad_width, iaxis, kwargs):
            # if no padding, skip directly
            if pad_width[0] == 0 and pad_width[1] == 0:
                return vector
            else:
                if all(vector[pad_width[0]:-pad_width[1]] == 0):
                    pad_value = 1
                else:
                    pad_value = 0

                vector[:pad_width[0]] = pad_value
                vector[-pad_width[1]:] = pad_value

        # adding silent note
        x = np.pad(x, ((0, 0), (0, 0), (0, 2), (0, 0)), mode=pad_with)

        # converting to categorical (keep only one note played at a time)
        tracks = []
        for t in range(midi_cfg.n_tracks):
            x_t = x[:, :, :, t]
            x_t = to_categorical(x_t.argmax(2), num_classes=midi_cfg.n_cropped_notes)
            x_t = np.expand_dims(x_t, axis=-1)

            tracks.append(x_t)

        x = np.concatenate(tracks, axis=-1)

        # adding held note
        for sample in range(x.shape[0]):
            for ts in range(1, x.shape[1]):
                for track in range(x.shape[3]):
                    # check for equality, except for the hold note position (the last position)
                    if np.array_equal(x[sample, ts, :-1, track], x[sample, ts - 1, :-1, track]):
                        x[sample, ts, -1, track] = 1

        # just zero the pianoroll where there is a held note
        for sample in range(x.shape[0]):
            for ts in range(1, x.shape[1]):
                for track in range(x.shape[3]):
                    if x[sample, ts, -1, track] == 1:
                        x[sample, ts, :-1, track] = 0

        # finally, use [0, 1] interval for ground truth Y and [-1, 1] interval for input/teacher forcing X
        y = x.copy()
        x[x == 1] = 1
        x[x == 0] = -1

        return x, y

    def create_samples(self, pianorolls: List[str]) -> List[str]:
        samples = []

        cs = h.get_chunksize(pianorolls)
        with mp.Pool(processes=cfg.processes) as pool:
            log.info("Creating samples out of {} songs...".format(len(pianorolls)))
            log.info("Number of processes: {}".format(cfg.processes))
            log.info("Chunksize: {}".format(cs))
            log.info("Number of chunks: {}".format(float(len(pianorolls) / cs)))
            for i, res in enumerate(pool.imap_unordered(iterable=pianorolls, func=self._create_sample, chunksize=cs)):
                if (i + 1) % cs == 0:
                    log.info("Processed {} chunks...".format(round(i / cs)))
                if res is not None:
                    samples.extend(res)

        log.info("{} samples have been created!".format(len(samples)))
        return samples

    def _create_sample(self, song: str) -> Optional[List[str]]:
        song = pproll.load(os.path.join(cfg.Paths.pianorolls, song))
        piano, others = get_instruments(song)

        combinations = list(itertools.product(piano, others))

        # Generate all combinations of our given tracks
        yeah = 0
        samples = []
        for i in combinations:
            mt = pproll.Multitrack()
            mt.remove_empty_tracks()

            for t in i:
                mt.append_track(
                    pianoroll=song.tracks[t].pianoroll,
                    program=song.tracks[t].program,
                    is_drum=song.tracks[t].is_drum,
                    name=song.tracks[t].name
                )
            mt.beat_resolution = song.beat_resolution
            mt.tempo = song.tempo

            mt.binarize()
            mt.assign_constant(1)

            # Check that no track is empty for the whole song
            if mt.get_empty_tracks():
                continue

            pianoroll = mt.get_stacked_pianoroll()

            for j in range(midi_cfg.phrase_size, pianoroll.shape[0] + 1, midi_cfg.phrase_size):
                window = pianoroll[j:j + midi_cfg.phrase_size, :, :]
                # print("window from", i, "to", i+self.phrase_size)

                # keep only the phrases that have at most one bar of consecutive silence
                # for each track
                # bar_of_silences = np.array([0] * midi_cfg.n_tracks)
                # for track in range(midi_cfg.n_tracks):
                #     j = 0
                #     while j + midi_cfg.bar_size <= window.shape[0]:
                #         if window[j:j + midi_cfg.bar_size, :, track].sum() == 0:
                #             bar_of_silences[track] += 1
                #
                #         j += 1  # self.bar_size

                # if the phrase is good, let's store it
                # if not any(bar_of_silences > 1):
                # data augmentation, random transpose bar
                shift = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], replace=False)
                tmp = pproll.Multitrack()
                tmp.remove_empty_tracks()
                for track in range(midi_cfg.n_tracks):
                    tmp.append_track(
                        pianoroll=window[:, :, track],
                        program=mt.tracks[track].program,
                        name=mt.tracks[track].name,
                        is_drum=mt.tracks[track].is_drum
                    )

                tmp.beat_resolution = midi_cfg.beat_resolution
                tmp.tempo = mt.tempo
                tmp.name = "{}_{}".format(song.name, yeah)

                tmp.transpose(shift)
                tmp.check_validity()
                pproll.save(os.path.join(cfg.Paths.samples, tmp.name), tmp)
                samples.append(tmp.name)
                yeah += 1
        return samples
