from glob import glob
from os.path import join
from random import random

import torchvision as tv
from torch import float32
from torch.utils.data import DataLoader

__all__ = ["Dataset", "DataLoader"]


class Dataset:
    def __init__(self):
        self.person = join("data", "person")
        self.cartoon = join("data", "cartoon")
        self.augment = tv.transforms.ColorJitter(0.1, 0.1, 0, 0.05)
        self.normalize = tv.transforms.Normalize([127.5] * 3, [127.5] * 3)

    def __len__(self):
        return len(glob(join(self.person, "*.png")))

    def __getitem__(self, item):
        person = self.load(join(self.person, f"{item}.png"))
        cartoon = self.load(join(self.cartoon, f"{item}.png"))
        if random() < 0.5:
            person = person.flip(-1)
            cartoon = cartoon.flip(-1)
        return tuple(map(
            lambda image: self.normalize(image).cuda(0),
            (self.augment(person), cartoon)
        ))

    @staticmethod
    def load(path):
        return tv.io.read_image(path).to(float32)
