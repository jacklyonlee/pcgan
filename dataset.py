import os
import numpy as np

import torch
import torch.utils.data

_synsetid_to_cate = {
    "02691156": "airplane",
    "02958343": "car",
    "03001627": "chair",
}
_cate_to_synsetid = {v: k for k, v in _synsetid_to_cate.items()}


class ShapeNet15k(torch.utils.data.Dataset):
    def __init__(self, root, cate, split, random_sample, sample_size):
        self.data = []
        cate_dir = os.path.join(root, _cate_to_synsetid[cate], split)
        for fname in os.listdir(cate_dir):
            if fname.endswith(".npy"):
                path = os.path.join(cate_dir, fname)
                sample = np.load(path)[np.newaxis, ...]
                self.data.append(torch.from_numpy(sample).float())

        # Normalize data
        self.data = torch.cat(self.data, dim=0)
        self.mu = self.data.view(-1, 3).mean(dim=0).view(1, 3)
        self.std = self.data.view(-1).std(dim=0).view(1, 1)
        self.data = (self.data - self.mu) / self.std

        # Following lines are purely for reproducing results of
        # the official SetVAE implementation: github.com/jw9730/setvae
        tr_data, te_data = self.data.split(10000, dim=1)
        self.data = tr_data if split == "train" else te_data

        self.random_sample = random_sample
        self.sample_size = sample_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        sample_idx = (
            torch.randperm(x.size(0))[: self.sample_size]
            if self.random_sample
            else torch.arange(self.sample_size)
        )
        x = x[sample_idx]
        return x, self.mu, self.std
