import os
import tempfile as tmp
from datetime import datetime as dt
from enum import Enum, auto
from os import path as p

import helper as h


class Exceptions(Enum):
    EXCEPTION_PARSING_PRETTY_MIDI = auto()
    EXCEPTION_PARSING_PYPIANOROLL = auto()
    WRONG_TRACK_COUNT = auto()


class Paths:
    resources = p.abspath(p.join(p.dirname(__file__), '..', 'res'))
    log = p.abspath(p.join(p.dirname(__file__), '..', 'log'))
    out = p.abspath(p.join(p.dirname(__file__), '..', 'out'))

    musae = p.abspath(os.path.join(tmp.gettempdir(), 'musae'))

    unzip = p.join(musae, 'unzip')
    plots = p.join(out, 'plots')
    checkpoints = p.join(out, 'checkpoints')
    generated = p.join(out, 'generated')

    h.create_dirs(musae)
    h.create_dirs(resources)
    h.create_dirs(log)
    h.create_dirs(out)
    h.create_dirs(plots)
    h.create_dirs(checkpoints)
    h.create_dirs(generated)


class Resources:
    dataset_zip = p.join(Paths.resources, 'lmd_filtered.zip')
    log_txt = p.join(Paths.log, "{}.txt".format(dt.now().strftime("%d-%m-%Y_%H.%M.%S")))
    best_encoder = p.join(Paths.checkpoints, "best_encoder.h5")
    best_decoder = p.join(Paths.checkpoints, "best_decoder.h5")


processes = 12
preprocess = True
batch_size = 128
