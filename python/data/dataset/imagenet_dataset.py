from __future__ import print_function

import numpy as np
import os

from .common_dataset import CommonDataset


class ImageNetDataset(CommonDataset):
    def _load_anno(self, seed=None):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(" ")
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(np.int64(l[1]))
                assert os.path.exists(self.images[-1])

class Cityscapes(CommonDataset):
    def _load_anno(self, seed=None):
        import glob
        assert os.path.exists(self._cls_path) # ground truth
        assert os.path.exists(self._img_root) # val dataset
        self.images = []
        self.labels = []

        self.labels = sorted(
            glob.glob(
                os.path.join(self._cls_path, '*', 
                             '*_gtFine_labelTrainIds.png')))[:5]
        self.images = sorted(
            glob.glob(
                os.path.join(self._img_root, '*', '*_leftImg8bit.png')))[:5]
