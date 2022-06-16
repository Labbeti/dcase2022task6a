#!/usr/bin/env python
# -*- coding: utf-8 -*-


def print_usage() -> None:
    print(
        "Usage :\n"
        "- Training a AAC model      : python -m dcase2022task6a.train [pl=PL] [data=DATA] [ARGS...]\n"
        "- Predict captions          : python -m dcase2022task6a.predict resume=RESUME input=INPUT\n"
        "- Download a dataset        : python -m dcase2022task6a.prepare [data=DATA] [ARGS...]\n"
        "- Print train options       : python -m dcase2022task6a.train --help\n"
        "- Print predict options     : python -m dcase2022task6a.predict --help\n"
        "- Print prepare options     : python -m dcase2022task6a.prepare --help\n"
        "- Print AAC version         : python -m dcase2022task6a.version\n"
    )


if __name__ == "__main__":
    print_usage()
