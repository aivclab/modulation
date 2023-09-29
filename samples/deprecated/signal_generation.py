#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15-01-2021
           """

from modulation.data.signal_generator import SignalGenerator
from scipy.signal import sweep_poly

if __name__ == "__main__":

    def asiuhda():
        """description"""
        from math import sin, cos
        from scipy.signal import square

        d = 1 / 10
        t = range(100)
        a = SignalGenerator(
            d,
            sin,
            cos,
            lambda x: cos(x + 1),
            lambda x: sin(x + 2),
            lambda x: cos(x + 3),
            lambda x: sin(x + 4),
            square,
        )
        from matplotlib import pyplot

        pyplot.plot([s for _, s in zip(t, a)])
        pyplot.show()

    def asduashdsdu():
        """

        :return:
        :rtype:
        """
        import numpy
        import scipy.signal

        sample_rate = 44100

        def sine_wave(hz, peak, n_samples=sample_rate):
            """Compute N samples of a sine wave with given frequency and peak amplitude.
            Defaults to one second.
            """
            length = sample_rate / float(hz)
            omega = numpy.pi * 2 / length
            xvalues = numpy.arange(int(length)) * omega
            onecycle = peak * numpy.sin(xvalues)
            return numpy.resize(onecycle, (n_samples,)).astype(numpy.int16)

        sum([sine_wave(440, 4096), sine_wave(880, 4096)])

    def asiuhda2():
        """description"""
        from math import sin, cos
        import numpy

        """
chirp(t, f0, t1, f1[, method, phi, vertex_zero]) 	#Frequency-swept cosine generator.
sweep_poly(t, poly[, phi]) 	#Frequency-swept cosine generator, with a time-dependent frequency.
"""
        t = numpy.arange(100) / 1000
        a = SignalGenerator(
            lambda x: sin(2 * numpy.pi * 50 * x),
            lambda x: cos(2 * numpy.pi * 20 * x),
        )
        from matplotlib import pyplot

        pyplot.plot(a(t))
        pyplot.show()

    def pygame_ex2():
        """description"""
        import numpy
        import pygame

        sample_rate = 44100
        freq = 440

        pygame.mixer.init(44100, -16, 2, 512)
        # sampling frequency, size, channels, buffer

        # Sampling frequency
        # Analog audio is recorded by sampling it 44,100 times per second,
        # and then these samples are used to reconstruct the audio signal
        # when playing it back.

        # size
        # The size argument represents how many bits are used for each
        # audio sample. If the value is negative then signed sample
        # values will be used.

        # channels
        # 1 = mono, 2 = stereo

        # buffer
        # The buffer argument controls the number of internal samples
        # used in the sound mixer. It can be lowered to reduce latency,
        # but sound dropout may occur. It can be raised to larger values
        # to ensure playback never skips, but it will impose latency on sound playback.

        arr = numpy.array(
            [
                4096 * numpy.sin(2.0 * numpy.pi * freq * x / sample_rate)
                for x in range(0, sample_rate)
            ]
        ).astype(numpy.int16)
        arr2 = numpy.c_[arr, arr]
        sound = pygame.sndarray.make_sound(arr2)
        sound.play(-1)
        pygame.time.delay(1000)
        sound.stop()

    def asijasd():
        """description"""
        t = numpy.linspace(0, 10, 5001)
        w = chirp(t, f0=12.5, f1=2.5, t1=10, method="linear")

        p = numpy.poly1d([0.05, -0.75, 2.5, 5.0])
        t = numpy.linspace(0, 10, 5001)
        w = sweep_poly(t, p)

    def pygame_ex1():
        """

        :return:
        :rtype:
        """
        import pygame, pygame.sndarray
        import numpy
        import scipy.signal
        from time import sleep

        sample_rate = 48000
        pygame.mixer.pre_init(sample_rate, -16, 1, 1024)
        pygame.init()

        def square_wave(hz, peak, duty_cycle=0.5, n_samples=sample_rate):
            """

            :param hz:
            :type hz:
            :param peak:
            :type peak:
            :param duty_cycle:
            :type duty_cycle:
            :param n_samples:
            :type n_samples:
            :return:
            :rtype:
            """
            t = numpy.linspace(0, 1, 500 * 440 / hz, endpoint=False)
            wave = scipy.signal.square(2 * numpy.pi * 5 * t, duty=duty_cycle)
            wave = numpy.resize(wave, (n_samples,))
            return peak / 2 * wave.astype(numpy.int16)

        def audio_freq(freq=800):
            """

            :param freq:
            :type freq:
            """
            global sound
            sample_wave = square_wave(freq, 4096)
            sound = pygame.sndarray.make_sound(sample_wave)

        # TEST
        audio_freq()
        sound.play(-1)
        sleep(0.5)
        sound.stop()
        audio_freq(1000)
        # sleep(1)
        sound.play(-1)
        sleep(2)
        sound.stop()
        sleep(1)
        sound.play(-1)
        sleep(0.5)

    # asiuhda()
    asiuhda2()
