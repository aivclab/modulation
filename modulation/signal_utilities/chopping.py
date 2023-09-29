#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 7/8/22
           """

__all__ = [
    "chop_n",
    "chop_n_end",
    "chop_n_middle",
    "chop_n_middle_end",
    "chop_n_middle_start",
    "chop_n_start",
    "mask_chop_segments",
    "separate_mask_regions",
]

from typing import Sequence, Tuple

import numpy
from modulation.signal_utilities.segmentation import runs_of_non_zero


def chop_n(x: Sequence, n: int) -> Sequence:
    """

    :param x:
    :type x:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    return x[:n]


def chop_n_end(x: Sequence, n: int) -> Sequence:
    """

    :param x:
    :type x:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    return x[-n:]


def chop_n_middle(x: Sequence, n: int) -> Sequence:
    """

    :param x:
    :type x:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    return x[n:-n]


def chop_n_middle_end(x: Sequence, n: int) -> Sequence:
    """

    :param x:
    :type x:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    return x[n:-n][::-1]


def chop_n_middle_start(x: Sequence, n: int) -> Sequence:
    """

    :param x:
    :type x:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    return x[n:-n][::-1]


def chop_n_start(x: Sequence, n: int) -> Sequence:
    """

    :param x:
    :type x:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    return x[n:]


def mask_chop_segments(mask: Sequence, data: Sequence) -> Sequence:
    """
    discards rest of the data if vad is shorter than data

    returns consecutive segments is where the mask is 1

    :param mask:
    :type mask:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    data_masked = [0] * len(data)

    for n in range(min(len(data), len(mask))):
        if mask[n] != 0:
            data_masked[n] = data[n]
        else:
            data_masked[n] = 0

    return [list(r) for b, r in runs_of_non_zero(data_masked) if b]


def separate_mask_regions(data: Sequence, mask: Sequence) -> Tuple[Sequence, Sequence]:
    m0 = numpy.concatenate(([False], mask, [False]))
    idx = numpy.flatnonzero(m0[1:] != m0[:-1])
    return [data[idx[i] : idx[i + 1]] for i in range(0, len(idx), 2)], [
        (idx[i], idx[i + 1]) for i in range(0, len(idx), 2)
    ]


if __name__ == "__main__":

    def asudh():
        c = numpy.zeros(10)
        c[1:5] = 1
        c[8:10] = 1

        s = list(range(10))
        print(c, s)

        print(mask_chop_segments(c, s))

    def uahsd():
        a = numpy.random.random((5, 2))
        m = numpy.array([True, True, False, True, True])

        print(separate_mask_regions(a, m))

    # asudh()
    uahsd()
