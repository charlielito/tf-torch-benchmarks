import typing as tp

import cv2
import numpy as np
import tensorflow as tf
import torch


def read_fn(filepath):
    with tf.io.gfile.GFile(filepath, "rb") as f:
        return f.read()


def read_image(filepath):
    buf = read_fn(filepath)
    buf = np.frombuffer(buf, np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    # arr = cv2.imread(filepath)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


class BasicImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return read_image(self.image_paths[i])


def get_dataloader(
    image_paths: str,
    num_workers: int = 0,
    batch_size: int = 32,
    shuffle: bool = True,
    prefetch_factor: tp.Optional[str] = 4,
) -> torch.utils.data.DataLoader:
    dataset = BasicImageDataset(image_paths)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
        persistent_workers=False,
    )


class IterDataset(torch.utils.data.Dataset):
    def __init__(self, iter, length):
        super().__init__()
        self.iter = iter
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return next(self.iter)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    data_dir = "gs://tf-vs-torch/test-data/images/pokemon_jpg"
    data_dir = "data/pokemon_jpg"

    image_paths = [
        os.path.join(data_dir, file) for file in tf.io.gfile.listdir(data_dir)
    ]
    # print(image_paths)
    dataset = get_dataloader(
        image_paths[:100],
        num_workers=4,
        batch_size=32,
        shuffle=False,
        prefetch_factor=2,
    )
    for batch in dataset:
        print(batch.shape)
        plt.imshow(batch[0])
        plt.show()
    for batch in dataset:
        print(batch.shape)
