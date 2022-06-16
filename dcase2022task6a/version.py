#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys

import pytorch_lightning
import torch
import yaml

import dcase2022task6a


def get_packages_versions() -> dict[str, str]:
    return {
        "dcase2022task6a": dcase2022task6a.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "pytorch": str(torch.__version__),
        "pytorch_lightning": pytorch_lightning.__version__,
    }


def main_version(*args, **kwargs) -> None:
    """Print some packages versions."""
    versions = get_packages_versions()
    print(yaml.dump(versions, sort_keys=False))


if __name__ == "__main__":
    main_version()
