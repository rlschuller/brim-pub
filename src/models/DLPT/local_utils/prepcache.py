import argparse
import sys
import os

import h5py

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

sys.path.append('./src')
sys.path.append('./src/models/DLPT')
from utils.util import enumerateWithEstimate

from dataload import MRIDataset
from utils.logconf import logging
from model import MRIModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

BREAST_MRI_DIR = './'
HDF_PATH = BREAST_MRI_DIR + '/data/processed/hdf5/'

class MRIPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        dataset = h5py.File('/tmp/test/cubes32stride16.hdf5', 'r')
        self.prep_dl = DataLoader(
            MRIDataset(dataset),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass
            
        log.info("Done")
        dataset.close()


if __name__ == '__main__':
    MRIPrepCacheApp().main()