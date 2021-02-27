import itertools
import json
import logging as log
import os
import pprint
import random
from collections import Counter
from typing import List

import numpy as np
import pretty_midi as pm
import pypianoroll as pproll
from keras.utils import to_categorical

import config as cfg
from config import midi_params

log.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


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


def count_genres(dataset_path, max_genres):
    # max_pbc = sum([len(files) for _, _, files in os.walk(os.path.join(dataset_path, "songs"))])
    # assign unique id for each song of dataset
    print("Extracting real song names...")
    pbc = 0
    counter = Counter()
    for path, subdirs, files in os.walk(os.path.join(dataset_path, "songs")):
        for song in files:
            pbc += 1

            song_number = song.split(".")[0]

            with open(os.path.join(dataset_path, "metadata", song_number + ".json")) as metadata_fp:
                metadata = json.load(metadata_fp)

            counter.update(metadata["genres"])

    print("Genres found:")
    pp.pprint(counter.most_common(max_genres))

    with open(os.path.join(dataset_path, "genre_counter.json"), "w") as fp:
        json.dump(counter.most_common(max_genres), fp)

    genres_list = [x[0] for x in list(counter.most_common(max_genres))]

    if not os.path.exists(os.path.join(dataset_path, "labels")):
        os.makedirs(os.path.join(dataset_path, "labels"))

    # now generate labels information (S latents)
    print("Generating labels information...")
    pbc = 0
    for path, subdirs, files in os.walk(os.path.join(dataset_path, "songs")):
        for song in files:
            pbc += 1

            song_number = song.split(".")[0]

            with open(os.path.join(dataset_path, "metadata", song_number + ".json")) as metadata_fp:
                metadata = json.load(metadata_fp)

            # setting corresponding tags
            label = np.zeros(max_genres)
            for genre in metadata["genres"]:
                try:
                    idx = genres_list.index(genre)
                    label[idx] = 1
                except ValueError:
                    pass

            np.save(os.path.join(dataset_path, "labels", song_number), label)


class MidiDataset:
    def __init__(self):
        # self.dataset_path = cfg.general_params["dataset_path"]

        self.n_tracks = midi_params["n_tracks"]
        self.n_midi_programs = midi_params["n_midi_programs"]
        self.n_cropped_notes = midi_params["n_cropped_notes"]
        self.phrase_size = midi_params["phrase_size"]
        self.bar_size = midi_params["bar_size"]

        # Initialize
        self.dataset_length = -1

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

        for i, batch in enumerate(batches):
            dest = []
            for sample in batch:
                dest.append(sample.get_stacked_pianoroll())
            dest = np.array(dest)
            x, y = self.preprocess(dest)
            batches_x.append(x)
            batches_y.append(y)
        return [batches_x, batches_y]

    def preprocess(self, x: np.ndarray) -> [np.ndarray, np.ndarray]:
        # if silent timestep (all 0), then set silent note to 1, else set
        # silent note to 0
        def pad_with(vector, pad_width):
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
        count_exceptions = 0
        count_wrong_time_signature = 0
        count_fulfills_requirements = 0

        piano_count = 0
        strings_count = 0
        ensemble_count = 0
        others_count = 0

        samples = []
        yeah = 0
        for count, song in enumerate(songs):
            if (count + 1) % 1000 == 0:
                log.info("Processed {} songs...".format(count))
            try:
                pretty_midi = pm.PrettyMIDI(song)
            except:
                count_exceptions += 1
                continue

            if not check_four_fourth(pretty_midi):
                count_wrong_time_signature += 1
                continue

            del pretty_midi

            try:
                base_song = pproll.parse(song)
            except:
                count_exceptions += 1
                continue

            # Divide the songs in piano, string, ensemble and others
            instruments = get_instruments(base_song)

            # Save count for statistics
            piano_count += instruments[0]
            strings_count += instruments[1]
            ensemble_count += instruments[2]
            others_count += instruments[3]

            # Check if there are any instruments that we classify as others
            if instruments[3]:
                # If so, merge all these tracks into one
                base_song.merge_tracks(instruments[3], mode="max", program=57, name="Other", remove_merged=True)
                # Restore the order destroyed by the merging
                instruments = get_instruments(base_song)

            # Song has to have at least one piano, string, and ensemble track
            if (not instruments[0]) | (not instruments[1]) | (not instruments[2]):
                continue
            count_fulfills_requirements += 1
            instruments = list(itertools.product(*instruments))

            # Generate all combinations of our given tracks
            for i in instruments:
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
                            tmp.name = str(yeah)

                            tmp.transpose(shift)
                            tmp.check_validity()
                            samples.append(tmp)
                            # tmp.save(os.path.join(pianorolls_path, str(yeah) + ".npz"))
                            del tmp
                            yeah += 1

                    i += self.bar_size
                del song
            del base_song

        log.warning("{} songs have thrown an exception!".format(count_exceptions))
        log.warning("{} songs have the wrong time signature!".format(count_wrong_time_signature))
        log.info("{} songs had at least one piano track.".format(piano_count))
        log.info("{} songs had at least one strings track.".format(strings_count))
        log.info("{} songs had at least one ensemble track.".format(ensemble_count))
        log.info("{} songs had at least one others track.".format(others_count))
        log.info("{} have at least one piano, string, and ensemble track.".format(count_fulfills_requirements))
        log.info("Out of the remaining songs, {} samples have been create and saved to.".format(len(samples)))
        return samples