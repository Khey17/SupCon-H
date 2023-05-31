import os
import torch
from typing import Any, Callable, Optional
from PIL import Image
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from torchvision.datasets.folder import default_loader
from collections import defaultdict
from random import sample
import random


class FERPair(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None):
        mode = "train" if train else "test"
        data_dir = os.path.join(root, "FERPlus", mode)
        super().__init__(data_dir, transform=transform)
        self.images, self.labels = self._load_data()

    def _load_data(self):
        images = []
        labels = []

        for img_path, label in self.samples:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

        return images, labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target

    def __len__(self):
        return len(self.images)


class FERPair_true_label(ImageFolder):
    def __init__(self, root, train=True, transform=None):
        mode = "train" if train else "test"
        data_dir = os.path.join(root, "FERPlus", mode)
        super().__init__(data_dir, transform=transform)
        self.images, self.labels = self._load_data()
        
    def _load_data(self):
        images = []
        labels = []
        
        for img_path, label in self.samples:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            labels.append(label)
        
        return images, labels
    
    def _get_label_index(self):
        label_index = defaultdict(list)
        
        for idx, label in enumerate(self.labels):
            label_index[label].append(idx)
        
        return label_index
    
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        pos_1 = img
        pos_2_idx = random.choice(self._get_label_index()[target])
        pos_2 = self.transform(self.images[pos_2_idx])
        
        return pos_1, pos_2, target

    def __len__(self):
        return len(self.images)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
#train_transform.transforms.insert(0, RandAugment(1, 10))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def get_dataset(dataset_name, M_view, N_view, root='./data/', pair=True):
    if pair:
        if dataset_name == 'ferplus':
            train_data = FERPair(root=root, train=True, transform=train_transform)
            memory_data = FERPair(root=root, train=True, transform=test_transform)
            test_data = FERPair(root=root, train=False, transform=test_transform)

        elif dataset_name == 'ferpair_true_label':
            train_data = FERPair_true_label(root=root, train=True, transform=train_transform)
            memory_data = FERPair_true_label(root=root, train=True, transform=test_transform)
            test_data = FERPair_true_label(root=root, train=False, transform=test_transform)

        else:
            raise Exception('Invalid dataset name')
    else:
        if dataset_name in ['ferplus', 'ferpair_true_label']:
            train_data = FERPair(root=root, train=True, transform=train_transform)
            memory_data = FERPair(root=root, train=True, transform=test_transform)
            test_data = FERPair(root=root, train=False, transform=test_transform)
        else:
            raise Exception('Invalid dataset name')

    return train_data, memory_data, test_data



# class CIFAR10Pair(CIFAR10):
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             pos_1 = self.transform(img)
#             pos_2 = self.transform(img)
 
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return pos_1, pos_2, target


# class CIFAR100Pair_true_label(CIFAR100):
#     #dataloader where pairs of positive samples are randomly sampled from pairs
#     #of inputs with the same label. 
#     def __init__(self, root='../data', train=True, transform=None):
#         super().__init__(root=root, train=train, transform=transform)
#         def get_labels(i):
#             return [index for index in range(len(self)) if self.targets[index]==i]

#         self.label_index = [get_labels(i) for i in range(100)]

#     def __getitem__(self, index):
#         img1, target = self.data[index], self.targets[index]

#         index_example_same_label=sample(self.label_index[self.targets[index]],1)[0]
#         img2 = self.data[index_example_same_label]

#         img1 = Image.fromarray(img1)
#         img2 = Image.fromarray(img2)

#         if self.transform is not None:
#             pos_1 = self.transform(img1)
#             pos_2 = self.transform(img2)
 
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return pos_1, pos_2, target

# class CIFAR100Pair(CIFAR100):
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             pos_1 = self.transform(img)
#             pos_2 = self.transform(img)
 
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return pos_1, pos_2, target


# class STL10Pair(STL10):
#     def __getitem__(self, index):
#         img, target = self.data[index], self.labels[index]
#         img = Image.fromarray(np.transpose(img, (1, 2, 0)))

#         if self.transform is not None:
#             pos_1 = self.transform(img)
#             pos_2 = self.transform(img)

#         return pos_1, pos_2, target