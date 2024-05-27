import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TrainValDataset(Dataset):
    def __init__(
            self,
            mode,
            data_dir,
            gt,
            fraction,
            transform=None,
    ):

        self._items = []
        self._transform = transform
        self._gt = self.read_csv(gt)

        dir_list = os.listdir(data_dir)

        if mode == "train":
            img_pathes = dir_list[: int(fraction * len(dir_list))]
        elif mode == "val":
            img_pathes = dir_list[int(fraction * len(dir_list)):]

        for filename in img_pathes:
            self._items.append((os.path.join(data_dir, filename), self._gt[filename]))

    @staticmethod
    def read_csv(filename):
        res = {}
        with open(filename) as fhandle:
            next(fhandle)
            for line in fhandle:
                parts = line.rstrip('\n').split(',')
                coords = np.array([float(x) for x in parts[1:]], dtype='float64')
                res[parts[0]] = coords
        return res

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, label = self._items[index]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)

        if self._transform is None:
            self._transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(96, 96),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        kp = iter(label)
        keypoints = [point for point in zip(kp, kp)]

        transformed = self._transform(image=image, keypoints=keypoints)

        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']

        labels = [coord for point in transformed_keypoints for coord in point]

        return transformed_image, torch.Tensor(labels)


class InferDataset(Dataset):
    def __init__(
            self,
            data_dir,
            transform=None,
    ):
        self._items = []
        self._transform = transform

        dir_list = os.listdir(data_dir)

        for filename in dir_list:
            self._items.append((os.path.join(data_dir, filename), filename))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, img_name = self._items[index]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)

        src_size = image.shape

        if self._transform is None:
            self._transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(96, 96),
                ToTensorV2(),
            ])

        transformed = self._transform(image=image)
        transformed_image = transformed['image']

        return transformed_image, img_name, src_size
