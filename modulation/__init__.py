#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
from warnings import warn

import pkg_resources
from apppath import AppPath

__project__ = "Modulation"
__author__ = "Christian Heider Nielsen"
__version__ = "0.0.4"
__doc__ = r"""
.. module:: modulation
   :platform: Unix, Windows
   :synopsis: A set of tools build for general signal processing.

.. moduleauthor:: Christian Heider Nielsen <christian.heider@alexandra.dk>

Created on 27/04/2019

@author: cnheider
"""
__all__ = [
    "PROJECT_APP_PATH",
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "get_version",
    "PROJECT_ORGANISATION",
    "PROJECT_AUTHOR",
    "PROJECT_YEAR",
    "INCLUDE_PROJECT_READMES",
    "PACKAGE_DATA_PATH",
]

from typing import Any
from pathlib import Path


def dist_is_editable(dist: Any) -> bool:
    """
    Return True if given Distribution is an editable installation."""
    import sys

    for path_item in sys.path:
        egg_link = Path(path_item) / f"{dist.project_name}.egg-link"
        if egg_link.is_file():
            return True
    return False


PROJECT_NAME = __project__.lower().strip().replace(" ", "_")
PROJECT_VERSION = __version__
PROJECT_YEAR = 2018
PROJECT_AUTHOR = __author__.lower().strip().replace(" ", "_")
PROJECT_ORGANISATION = "Aivclab"
PROJECT_APP_PATH = AppPath(app_name=PROJECT_NAME, app_author=PROJECT_AUTHOR)
PACKAGE_DATA_PATH = Path(pkg_resources.resource_filename(PROJECT_NAME, "data"))
INCLUDE_PROJECT_READMES = False

distributions = {v.key: v for v in pkg_resources.working_set}
if PROJECT_NAME in distributions:
    distribution = distributions[PROJECT_NAME]
    DEVELOP = dist_is_editable(distribution)
else:
    DEVELOP = True


def get_version(append_time: Any = DEVELOP) -> str:
    """

    :param append_time:
    :type append_time:
    :return:
    :rtype:
    """
    version = __version__
    if not version:
        version = os.getenv("VERSION", "0.0.0")

    if append_time:
        now = datetime.datetime.utcnow()
        date_version = now.strftime("%Y%m%d%H%M%S")
        # date_version = time.time()

        if version:
            # Most git tags are prefixed with 'v' (example: v1.2.3) this is
            # never desirable for artifact repositories, so we strip the
            # leading 'v' if it's present.
            version = (
                version[1:]
                if isinstance(version, str) and version.startswith("v")
                else version
            )
        else:
            # Default version is an ISO8601 compliant datetime. PyPI doesn't allow
            # the colon ':' character in its versions, and time is required to allow
            # for multiple publications to master in one day. This datetime string
            # uses the 'basic' ISO8601 format for both its date and time components
            # to avoid issues with the colon character (ISO requires that date and
            # time components of a date-time string must be uniformly basic or
            # extended, which is why the date component does not have dashes.
            #
            # Publications using datetime versions should only be made from master
            # to represent the HEAD moving forward.
            warn(
                f"Environment variable VERSION is not set, only using datetime: {date_version}"
            )

            # warn(f'Environment variable VERSION is not set, only using timestamp: {version}')

        version = f"{version}.{date_version}"

    return version


if __version__ is None:
    __version__ = get_version(append_time=True)

__version_info__ = tuple(int(segment) for segment in __version__.split("."))

if __name__ == "__main__":
    print(PROJECT_APP_PATH.user_log)
