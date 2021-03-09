from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re

from ..utils.data import BaseImageDataset


class dukeCam(BaseImageDataset):
    dataset_dir = '.'

    def __init__(self, root):
        super(dukeCam, self).__init__()
        self.dataset_dir = root
        train = self._process_dir(self.dataset_dir, relabel=True)
        self.train = train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # index and their corres pid

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
