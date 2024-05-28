from argparse import ArgumentParser

import os
from pathlib import Path
from os import fspath
import sys

project_path = Path()
sys.path.append(fspath(project_path / 'utils'))

import train, infer


def get_args():
    parser = ArgumentParser(description='CLI')
    parser.add_argument('-i', '-infer', '-inference',
                        action='store_true', default=False, required=False,
                        help='Flag to use model for inference instead of training it')

    args = parser.parse_args()
    return args


def main():
    os.chdir('utils')
    if get_args().i:
        print('Model is in inference mode\n')
        infer.main()
    else:
        print('Model is in training mode\n')
        train.main()


if __name__ == "__main__":
    main()
