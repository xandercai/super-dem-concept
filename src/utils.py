import os
import yaml
from dotenv import load_dotenv
from easydict import EasyDict as edict
import json

# Load the stored environment variables
load_dotenv()
#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            if not os.path.exists(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
                self.file = open(fpath, "w")
            else:
                self.file = open(fpath, "a")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def get_env_variable(var_name: str, allow_empty: bool = False) -> str:
    """
    Get the environment variable or raise an exception
    """
    env_var = os.getenv(var_name)
    if not allow_empty and env_var in (None, ""):
        raise KeyError(
            f"Environment variable {var_name} not set, and allow_empty is False"
        )
    return env_var


def create_config(file, verbose: bool = True):
    # Read the files
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    if verbose:
        print(json.dumps(cfg, indent=4))

    return cfg


if __name__ == "__main__":
    create_config("./config/dem_nz/deeplabv3_resnet50/semseg.yml")
