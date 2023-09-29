#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10-12-2020
           """

from pathlib import Path
from typing import Any

import numpy
from draugr.numpy_utilities import root_mean_square
from draugr.visualisation import progress_bar
from modulation.signal_utilities import mask_split_non_zero_concat
from scipy import signal
from scipy.io import wavfile
from warg import ensure_existence

__all__ = ["sample_noise", "compute_additive_noise_samples"]


def sample_noise(
    noise,
    noise_rate: int,
    signal_len: int,
    signal_rate: int,
    *,
    resample_noise: bool = True,
) -> Any:
    """

    :param noise:
    :type noise:
    :param noise_rate:
    :type noise_rate:
    :param signal_len:
    :type signal_len:
    :param signal_rate:
    :type signal_rate:
    :param resample_noise:
    :type resample_noise:
    :return:
    :rtype:
    """
    noise_len = len(noise)
    if resample_noise:
        noise = signal.resample(
            noise, round(noise_len * float(signal_rate) / noise_rate)
        )

    start_index = numpy.random.randint(0, noise_len)
    noise = numpy.tile(noise, (signal_len // noise_len) + 2)  # at least tile once (=2)
    return noise[start_index : signal_len + start_index]


def compute_additive_noise_samples(
    *,
    voice_activity_mask: numpy.ndarray,
    signal_file: Path,
    category,
    out_dir,
    noise_file,
    snrs=None,
    verbose: bool = False,
) -> None:
    """

    :param voice_activity_mask:
    :type voice_activity_mask:
    :param signal_file:
    :type signal_file:
    :param category:
    :type category:
    :param out_dir:
    :type out_dir:
    :param noise_file:
    :type noise_file:
    :param snrs:
    :type snrs:
    :param verbose:
    :type verbose:
    """
    if snrs is None:
        snrs = list((i * 5 for i in range(5)))

    sr_noise, noise = wavfile.read(str(noise_file))
    sr_signal, signal = wavfile.read(str(signal_file))

    max_sample = numpy.max(signal)
    signal = signal / max_sample

    noise_part = sample_noise(
        noise / numpy.max(noise),
        noise_rate=sr_noise,
        signal_len=len(signal),
        signal_rate=sr_signal,
    )

    noise_scaled = noise_part * (
        root_mean_square(mask_split_non_zero_concat(voice_activity_mask, signal)[0])
        / root_mean_square(noise_part)
    )  # scaled by ratio of speech to noise level

    for snr in progress_bar(snrs, disable=not verbose):
        noised = signal + noise_scaled / (10 ** (snr / 20))
        dest = str(
            ensure_existence(
                out_dir / f'{noise_file.with_suffix("").name}_SNR_{snr}dB' / category
            )
            / signal_file.name
        )
        if verbose:
            print(f"writing wavfile {dest}")
        wavfile.write(
            dest,
            sr_signal,
            ((noised / numpy.max(noised)) * max_sample).astype(numpy.int16),
        )


if __name__ == "__main__":
    pass
