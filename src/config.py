import tempfile as tmp
from datetime import datetime as dt
from enum import Enum, auto
from os import path as p

import helper as h


class Exceptions(Enum):
    EXCEPTION_PARSING_PRETTY_MIDI = auto()
    EXCPETION_PARSING_PYPIANOROLL = auto()
    WRONG_TRACK_COUNT = auto()


class Paths:
    resources = p.abspath(p.join(p.dirname(__file__), '..', 'res'))
    log = p.abspath(p.join(p.dirname(__file__), '..', 'log'))
    out = p.abspath(p.join(p.dirname(__file__), '..', 'out'))

    pianorolls = p.abspath(p.join(tmp.gettempdir(), 'pianorolls'))
    samples = p.abspath(p.join(tmp.gettempdir(), 'samples'))
    batches = p.abspath(p.join(tmp.gettempdir(), 'batches'))
    unzip = p.abspath(p.join(tmp.gettempdir(), 'unzip'))
    x = p.abspath(p.join(batches, 'x'))
    y = p.abspath(p.join(batches, 'y'))
    plots = p.abspath(p.join(out, 'plots'))
    checkpoints = p.abspath(p.join(out, 'checkpoints'))

    h.create_dirs(resources)
    h.create_dirs(log)
    h.create_dirs(batches)
    h.create_dirs(pianorolls)
    h.create_dirs(samples)
    h.create_dirs(x)
    h.create_dirs(y)
    h.create_dirs(out)
    h.create_dirs(plots)
    h.create_dirs(checkpoints)


class Resources:
    dataset_zip = p.abspath(p.join(Paths.resources, 'dataset.zip'))
    log_txt = p.join(Paths.log, "{}.{}".format(dt.now().strftime("%d-%m-%Y_%H.%M.%S"), "txt"))


processes = 12
preprocess = True
batch_size = 256
