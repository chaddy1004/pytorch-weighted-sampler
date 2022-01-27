import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from collections import defaultdict

random.seed(a=1004)
np.random.seed(seed=1004)
torch.manual_seed(1004)

if torch.cuda.is_available():
    torch.cuda.manual_seed(1004)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VanillaDataset(Dataset):
    def __init__(self, ratio, len_normal=100) -> None:
        super().__init__()

        self.positive_labels = [1 for _ in range(len_normal * ratio)]
        self.negative_labels = [0 for _ in range(len_normal)]

        self.n_positives = len(self.positive_labels)
        self.n_negatives = len(self.negative_labels)

        self.shuffled_labels = self.positive_labels + self.negative_labels

        random.shuffle(self.shuffled_labels)

    def __getitem__(self, index):
        return self.shuffled_labels[index]

    def __len__(self):
        return len(self.shuffled_labels)


class OverSampler(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sampler = None
        self.batch_size = 1
        self.n_samples = len(self.dataset)
        self.replacement = True

        self.final_size = self.dataset.n_positives * 2

        self.init_sampler()

        self.shuffle = self.sampler is None

        super().__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0,
                         sampler=self.sampler)

    def init_sampler(self):
        # class 0 : 43200개, class 1 : 4800개
        labels = self.dataset.shuffled_labels

        class_counts = {0: self.dataset.n_negatives, 1: self.dataset.n_positives}
        class_weights = defaultdict()
        for label in class_counts.keys():
            class_weights[label] = self.n_samples / class_counts[label]

        print(f"Class Weights: {class_weights}")

        weights = [class_weights[labels[i]] for i in range(self.n_samples)]
        self.sampler = WeightedRandomSampler(torch.DoubleTensor(weights), self.final_size, replacement=self.replacement)


class UnderSampler(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sampler = None
        self.batch_size = 1
        self.n_samples = len(self.dataset)
        self.replacement = False

        # self.final_size = self.dataset.n_negatives * 2
        self.final_size = self.dataset.n_negatives * 2

        self.init_sampler()

        self.shuffle = self.sampler is None

        super().__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0,
                         sampler=self.sampler)

    def init_sampler(self):
        # class 0 : 43200개, class 1 : 4800개
        labels = self.dataset.shuffled_labels

        class_counts = {0: self.dataset.n_negatives, 1: self.dataset.n_positives}
        class_weights = defaultdict()
        for label in class_counts.keys():
            class_weights[label] = self.n_samples / class_counts[label]

        print(f"Class Weights: {class_weights}")

        weights = [class_weights[labels[i]] for i in range(self.n_samples)]
        self.sampler = WeightedRandomSampler(torch.DoubleTensor(weights), self.final_size, replacement=self.replacement)


class NormalSampler(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sampler = None
        self.batch_size = 1
        self.n_samples = len(self.dataset)
        self.replacement = False

        self.final_size = self.n_samples

        self.shuffle = self.sampler is None

        super().__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0,
                         sampler=self.sampler)

    def init_sampler(self):
        # class 0 : 43200개, class 1 : 4800개
        labels = self.dataset.shuffled_labels

        class_counts = {0: self.dataset.n_negatives, 1: self.dataset.n_positives}
        class_weights = defaultdict()
        for label in class_counts.keys():
            class_weights[label] = self.n_samples / class_counts[label]

        print(f"Class Weights: {class_weights}")

        weights = [class_weights[labels[i]] for i in range(self.n_samples)]
        self.sampler = WeightedRandomSampler(torch.DoubleTensor(weights), self.final_size, replacement=self.replacement)