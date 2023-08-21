# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

from functools import reduce
from typing import Callable, Iterable, Sequence, Union

import numpy
from warg import Number, identity

__all__ = ["SignalGenerator", "multifreq", "triangle", "sawtooth"]


def multifreq(x: numpy.ndarray) -> numpy.ndarray:
    return (
        2
        + numpy.sin(x * numpy.pi)
        + 0.5 * numpy.sin(2 * x * numpy.pi)
        - 0.2 * numpy.cos(5 * x * numpy.pi)
    )


def triangle(x: numpy.ndarray, section_length: float = 0.5) -> numpy.ndarray:
    section0 = x < section_length
    section1 = (x >= section_length) & (x < 2 * section_length)
    section2 = (x >= 2 * section_length) & (x < 3 * section_length)
    section3 = x >= 3 * section_length
    output = numpy.zeros_like(x)
    output[section0] = x[section0]
    output[section1] = 2 * section_length - x[section1]
    output[section2] = x[section2] - 2 * section_length
    output[section3] = 4 * section_length - x[section3]
    return output


def sawtooth(x: numpy.ndarray, section_length: float = 0.5) -> numpy.ndarray:
    section0 = x < section_length
    section1 = (x >= section_length) & (x < 2 * section_length)
    section2 = (x >= 2 * section_length) & (x < 3 * section_length)
    section3 = x >= 3 * section_length
    output = numpy.zeros_like(x)
    output[section0] = x[section0]
    output[section1] = x[section1] - section_length
    output[section2] = x[section2] - 2 * section_length
    output[section3] = x[section3] - 3 * section_length
    return output


class SignalGenerator:
    """description"""

    def __init__(self, *funcs: Union[Callable, Number], delta_time: float = 1.0):
        self.reset_internal_time()
        self.delta_time = delta_time
        self.funcs = (0, *funcs)

    def __iter__(self):
        self.reset_internal_time()
        return self

    def __next__(self):
        self.t += self.delta_time
        return self.apply(self.t)

    def apply(self, t: float) -> float:
        """

        :param t:
        :type t:
        :return:
        :rtype:
        """
        return reduce(lambda x, y: x + y(t), self.funcs)

    def __call__(self, t: Iterable[Number]) -> Sequence:
        return [self.apply(i) for i in t]

    def reset_internal_time(self):
        """description"""
        self.t = 0.0

    def set_internal_time(self, t):
        """

        :param t:
        :type t:
        """
        self.t = t

    def __enter__(self):
        self.reset_internal_time()
        return True


if __name__ == "__main__":

    def asidjashdya():
        """
        counts
        """
        for _, i in zip(range(10), SignalGenerator(identity)):
            print(i)

    asidjashdya()
