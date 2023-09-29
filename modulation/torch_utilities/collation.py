#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-12-2020
           """

from typing import Iterable, Tuple

import torch
from draugr.torch_utilities import global_torch_device
from modulation.torch_utilities.padding import min_length_pad, pad_sequence
from torch.types import Device

__all__ = ["collate_pad_wrapped", "collate_transform_wrapped"]

from matplotlib import pyplot
import numpy
import wave


# shows the sound waves
def visualize(path: str):
    """

    :param path:
    :type path:
    """
    # reading the audio file
    raw = wave.open(path)

    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = numpy.frombuffer(signal, dtype="int16")

    # gets the frame rate
    f_rate = raw.getframerate()

    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time = numpy.linspace(0, len(signal) / f_rate, num=len(signal))

    pyplot.figure(1)
    pyplot.title("Sound Wave")
    pyplot.xlabel("Time")
    pyplot.plot(time, signal)
    pyplot.show()


def collate_pad_wrapped(device: Device = global_torch_device()) -> callable:
    """

    :param device:
    :type device:
    :return:
    :rtype:
    """

    def collate_fn(batch):
        """
        Pads batch of variable length

        returns batch, lengths, masks

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        """

        batch_ = torch.nn.utils.rnn.pad_sequence(
            [torch.Tensor(t).to(device) for t in batch]
        )

        return (
            batch_,
            torch.tensor([t.shape[0] for t in batch]).to(
                device
            ),  # get original sequence lengths
            (batch_ != 0).to(device),  # compute mask
        )

    return collate_fn


def collate_transform_wrapped(
    mapping_func: callable,
    transform: callable,
    min_length: int = 16000,
    device: Device = global_torch_device(),
) -> callable:
    """

    :param mapping_func:
    :type mapping_func:
    :param transform:
    :type transform:
    :param min_length:
    :type min_length:
    :param device:
    :type device:
    :return:
    :rtype:
    """

    def collate_fn(batch: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        To turn a list of data point made of audio recordings and utterances
        into two batched tensors for the model, we implement a collate function
        which is used by the PyTorch DataLoader that allows us to iterate over a
        dataset by batches. Please see `the
        documentation <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`__
        for more information about working with a collate function.

        In the collate function, we also apply the resampling, and the text
        encoding.


        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        :param batch:
        :return:
        """

        tensors, targets = [], []

        for (
            waveform,
            _,
            label,
            *_,
        ) in batch:  # Gather in lists, and encode labels as indices
            tensors += [min_length_pad(waveform, min_length)]
            targets += [mapping_func(label)]

        return (
            transform(pad_sequence(tensors).to(device)),
            torch.stack(targets).to(device),
        )

    return collate_fn
