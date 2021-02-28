import itertools
import logging as log
import multiprocessing as mp
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi as pm
import pypianoroll as pproll
from keras.utils import to_categorical

import config as cfg
from config import midi_params

log.getLogger(__name__)


def get_chunks(samples: List[pproll.Multitrack], n: int) -> List[List[pproll.Multitrack]]:
    for i in range(0, len(samples), n):
        yield samples[i:i + n]


def get_instruments(song: pproll.Multitrack) -> [list, list, list, list]:
    piano = []
    string = []
    ensemble = []
    other = []

    for i, track in enumerate(song.tracks):
        if track.program in range(8):
            track.name = "Piano"
            piano.append(i)
        elif track.program in range(40, 48):
            track.name = "Strings"
            string.append(i)
        elif track.program in range(48, 56):
            track.name = "Ensemble"
            ensemble.append(i)
        else:
            track.name = "Others"
            other.append(i)

    return piano, string, ensemble, other


def check_four_fourth(song: pm.PrettyMIDI):
    return all([tmp.numerator == 4 and tmp.denominator == 4 for tmp in song.time_signature_changes])


class MidiDataset:
    def __init__(self):
        # self.dataset_path = cfg.general_params["dataset_path"]

        self.n_tracks = midi_params["n_tracks"]
        self.n_midi_programs = midi_params["n_midi_programs"]
        self.n_cropped_notes = midi_params["n_cropped_notes"]
        self.phrase_size = midi_params["phrase_size"]
        self.bar_size = midi_params["bar_size"]

        # Initialize

    def select_batch(self, idx):
        x = np.load(os.path.join(self.dataset_path, "batches", "X", str(idx) + ".npy"))
        y = np.load(os.path.join(self.dataset_path, "batches", "Y", str(idx) + ".npy"))
        label = np.load(os.path.join(self.dataset_path, "batches", "labels", str(idx) + ".npy"))
        return x, y, label

    # warning: tends to use a lot of storage (disk) space
    def create_batches(self, samples: List[pproll.Multitrack], batch_size=cfg.batch_size) -> List[list]:
        random.shuffle(samples)

        batches = list(get_chunks(samples, batch_size))

        log.info("Number of samples to create batches of: {}".format(len(samples)))
        log.info("Batch size: {}".format(batch_size))
        log.info("Number of batches: {}".format(len(batches)))

        batches_x = []
        batches_y = []

        chunksize = int((len(batches) / cfg.processes) / cfg.processes) + 1
        with mp.Pool(processes=cfg.processes) as pool:
            count = 1
            for res in pool.imap_unordered(iterable=batches, func=self._process_batch, chunksize=chunksize):
                if count % 1000 == 0:
                    log.info("Processed {} batches...".format(count))
                batches_x.append(res[0])
                batches_y.append(res[1])
                count += 1
            pool.close()
            pool.join()

        return [batches_x, batches_y]

    def _process_batch(self, batch: List[pproll.Multitrack]):
        dest = []
        for sample in batch:
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
        for t in range(self.n_tracks):
            x_t = x[:, :, :, t]
            x_t = to_categorical(x_t.argmax(2), num_classes=self.n_cropped_notes)
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

    def create_samples(self, songs: list) -> List[pproll.Multitrack]:
        usable_songs = []

        samples = []
        chunksize = int((len(songs) / cfg.processes) / cfg.processes) + 1

        with mp.Pool(processes=cfg.processes) as pool:
            for res in pool.imap_unordered(iterable=songs, func=self._filter_out_song, chunksize=chunksize):
                if not res:
                    usable_songs.append(res)

            log.info("{} songs fulfill our requirements".format(len(usable_songs)))

            count = 0
            for res in pool.imap_unordered(iterable=usable_songs, func=self._create_sample, chunksize=chunksize):
                count += 1
                if count % chunksize == 0:
                    log.info("Processed {} songs...".format(count))
            samples.extend(res)
            pool.close()
            pool.join()

        log.info("Out of the remaining songs, {} samples have been created.".format(len(samples)))
        return samples

    def _filter_out_song(self, song: str) -> bool:
        try:
            pretty_midi = pm.PrettyMIDI(song)
        except:
            return True

        if not check_four_fourth(pretty_midi):
            return True

        del pretty_midi

        try:
            base_song = pproll.parse(song)
        except:
            return True

        # Divide the songs in piano, string, ensemble and others
        instruments = get_instruments(base_song)

        # Check if there are any instruments that we classify as others
        if instruments[3]:
            # If so, merge all these tracks into one
            base_song.merge_tracks(instruments[3], mode="max", program=57, name="Other", remove_merged=True)
            # Restore the order destroyed by the merging
            instruments = get_instruments(base_song)

        if len(instruments) != 4:
            return True
        return False

    def _create_sample(self, song_path: str) -> List[pproll.Multitrack]:
        base_song = pproll.parse(song_path)
        instruments = get_instruments(base_song)
        combinations = list(itertools.product(*instruments))

        # Generate all combinations of our given tracks
        yeah = 0
        samples = []
        for i in combinations:
            song = pproll.Multitrack()
            song.remove_empty_tracks()

            for t in i:
                song.append_track(
                    pianoroll=base_song.tracks[t].pianoroll,
                    program=base_song.tracks[t].program,
                    is_drum=base_song.tracks[t].is_drum,
                    name=base_song.tracks[t].name
                )
            song.beat_resolution = base_song.beat_resolution
            song.tempo = base_song.tempo

            song.binarize()
            song.assign_constant(1)

            # Check that no track is empty for the whole song
            if song.get_empty_tracks():
                continue

            pianoroll = song.get_stacked_pianoroll()

            i = 0
            while i + self.phrase_size <= pianoroll.shape[0]:
                window = pianoroll[i:i + self.phrase_size, :, :]
                # print("window from", i, "to", i+self.phrase_size)

                # keep only the phrases that have at most one bar of consecutive silence
                # for each track
                bar_of_silences = np.array([0] * self.n_tracks)
                for track in range(self.n_tracks):
                    j = 0
                    while j + self.bar_size <= window.shape[0]:
                        if window[j:j + self.bar_size, :, track].sum() == 0:
                            bar_of_silences[track] += 1

                        j += 1  # self.bar_size

                # if the phrase is good, let's store it
                if not any(bar_of_silences > 0):
                    # data augmentation, random transpose bar
                    for shift in np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], 1,
                                                  replace=False):
                        tmp = pproll.Multitrack()
                        tmp.remove_empty_tracks()
                        for track in range(self.n_tracks):
                            tmp.append_track(
                                pianoroll=window[:, :, track],
                                program=song.tracks[track].program,
                                name=cfg.instrument_names[song.tracks[track].program],
                                is_drum=song.tracks[track].is_drum
                            )

                        tmp.beat_resolution = 4
                        tmp.tempo = song.tempo
                        tmp.name = "{}_{}".format(song_path.split("\\")[-1], yeah)

                        tmp.transpose(shift)
                        tmp.check_validity()
                        samples.append(tmp)
                        # self.preprocess(np.array(tmp.get_stacked_pianoroll()))
                        del tmp
                        yeah += 1

                i += self.bar_size
            del song
        del base_song
        return samples
