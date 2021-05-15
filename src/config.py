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
    WRONG_SIGNATURE = auto()


class Paths:
    resources = p.abspath(p.join(p.dirname(__file__), '..', 'res'))
    log = p.abspath(p.join(p.dirname(__file__), '..', 'log'))
    out = p.abspath(p.join(p.dirname(__file__), '..', 'out'))

    musae = p.abspath(os.path.join(tmp.gettempdir(), 'musae'))

    unzip = p.join(musae, 'unzip')
    samples = p.join(musae, 'samples')
    batches = p.join(musae, 'batches')
    plots = p.join(out, 'plots')
    checkpoints = p.join(out, 'checkpoints')
    generated = p.join(out, 'generated')
    latent_space = p.join(plots, 'latent_space')

    b_samples = p.join(out, 'samples')
    b_batches = p.join(out, 'batches')

    h.create_dirs(b_batches)
    h.create_dirs(b_samples)
    h.create_dirs(batches)
    h.create_dirs(samples)
    h.create_dirs(musae)
    h.create_dirs(resources)
    h.create_dirs(log)
    h.create_dirs(out)
    h.create_dirs(plots)
    h.create_dirs(checkpoints)
    h.create_dirs(generated)
    h.create_dirs(latent_space)


class Resources:
    dataset_name = "kunstderfuge"
    dataset_zip = p.join(Paths.resources, f"{dataset_name}.zip")
    log_txt = p.join(Paths.log, "{}.txt".format(dt.now().strftime("%d-%m-%Y_%H.%M.%S")))
    best_encoder = p.join(Paths.checkpoints, f"{dataset_name}_best_encoder.h5")
    best_decoder = p.join(Paths.checkpoints, f"{dataset_name}_best_decoder.h5")


processes = 12
preprocess = True
batch_size = 16
