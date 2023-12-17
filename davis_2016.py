from __future__ import division

import numpy as np
from torch.utils.data import Dataset


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(
        self,
        train=True,
        img_list=[],
        labels=[],
        transform=None,
        meanval=(104.00699, 116.66877, 122.67892),
    ):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.transform = transform
        self.meanval = meanval

        if self.train:
            fname = "train_seqs"
        else:
            fname = "val_seqs"

        assert len(labels) == len(img_list)

        self.img_list = img_list
        self.labels = labels

        print("Done initializing " + fname + " Dataset")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {"image": img, "gt": gt}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = self.img_list[idx]
        if self.labels[idx] is not None:
            label = self.labels[idx]
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt / np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = self.img_list[0]
        return list(img.shape[:2])
