from __future__ import print_function

import numpy as np
import os

from .common_dataset import CommonDataset

class CityScapesDataset(CommonDataset):
    def _load_anno(self, seed=None):
        import glob
        assert os.path.exists(self._lab_path) # ground truth
        assert os.path.exists(self._img_root) # val dataset
        self.images = []
        self.labels = []

        self.labels = sorted(
            glob.glob(
                os.path.join(self._lab_path, '*', '*_gtFine_labelTrainIds.png')))
        self.images = sorted(
            glob.glob(
                os.path.join(self._img_root, '*', '*_leftImg8bit.png')))