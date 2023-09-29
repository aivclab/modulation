#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

from typing import Optional, Sequence

import numpy
from matplotlib import pyplot
from matplotlib.collections import LineCollection
from warg import Number

__all__ = ["color_plot"]


def color_plot(
    xy: numpy.ndarray,
    colors: Optional[Sequence[Number]] = None,
    cmap=pyplot.cm.gist_rainbow,  # cmap=pyplot.cm.gist_ncar
) -> None:
    """

    :param xy:
    :type xy:
    :param colors:
    :type colors:
    :param cmap:
    :type cmap:
    """
    fig, ax = pyplot.subplots()

    # Reshape things so that we have a sequence of:
    # [[(x0,y0),(x1,y1)],[(x0,y0),(x1,y1)],...]
    xy = xy.reshape(-1, 1, 2)
    segments = numpy.hstack([xy[:-1], xy[1:]])

    coll = LineCollection(segments, cmap=cmap)

    if colors is None:
        colors = numpy.arange(xy.shape[0])

    colors = (1 / xy.shape[0]) * colors
    # cyclic color = numpy.sin(colors)
    coll.set_array(colors)

    ax.add_collection(coll)
    ax.autoscale_view()


if __name__ == "__main__":

    def main2():
        """description"""
        xy = (numpy.random.random((1000, 2)) - 0.5).cumsum(axis=0)

        color_plot(xy)
        pyplot.show()

    def main():
        """description"""
        xy = (numpy.random.random((1000, 2)) - 0.5).cumsum(axis=0)

        color_plot(xy, numpy.random.random((1000)))
        pyplot.show()

    main()
